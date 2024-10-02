import os
import pandas as pd
import numpy as np
import math
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
from scipy.interpolate import LinearNDInterpolator

river_active = False  # 비활성 할꺼면 False 활성할꺼면 True
file_paths = r"C:\Users\user\Desktop\형기형 코드\파일"  
M_route = os.path.join(file_paths, "충적2p.txt")
S_route = os.path.join(file_paths, "암반2p.txt")
river_line = os.path.join(file_paths, "riverline.shp")

if river_active == True:
    R_route = os.path.join(file_paths, "river.txt")

X_p = 243610
Y_p = 509290
cell_s = 20      # 모델 격자 크기
sd_depth = 7.15
sd_depth2 = 12.5

river_depth_area1 = 100
river_depth_area2 = 200

iterative=40      #평활화 반복 횟수
admit_range=1.8  #최소 겹치기 범위
effect_cell_count = 20 #평활화 이동평균 참조 격자 영역

version = 0 # 0 = 평탄하게 , 1 = 표고따라 평활화 방법

# 초기 데이터프레임 생성
sd_pd = pd.DataFrame(columns=['X', 'Y', 'lower EL', 'upper EL'])
with open(M_route, "r") as sd:
    for num, line in enumerate(sd):
        if num > 0:  # 첫 번째 줄은 헤더일 가능성 있음, 건너뜀
            line = line.strip()
            parts = line.split('\t')
            sd_pd.loc[num-1, "X"] = float(parts[0])
            sd_pd.loc[num-1, "Y"] = float(parts[1])
            sd_pd.loc[num-1, "lower EL"] = float(parts[2]) - sd_depth
            sd_pd.loc[num-1, "upper EL"] = float(parts[2])

un_pd = pd.DataFrame(columns=['X', 'Y', 'EL'])
with open(S_route, "r") as sd:
    for num, line in enumerate(sd):
        if num > 0:
            line = line.strip()
            parts = line.split('\t')
            un_pd.loc[num-1, "X"] = float(parts[0])
            un_pd.loc[num-1, "Y"] = float(parts[1])
            un_pd.loc[num-1, "EL"] = float(parts[2])

if river_active == True:            
    river_pd = pd.read_csv(R_route, sep='\s+', engine='python')

river_gdf = gpd.read_file(river_line)

def river_cell_linear(buffer_distance1, buffer_distance2, river_gdf, sd_pd):
    geometry = [Point(xy) for xy in zip(sd_pd['X'], sd_pd['Y'])]
    gdf = gpd.GeoDataFrame(sd_pd, geometry=geometry)
    river_gdf['buffered_1'] = river_gdf['geometry'].buffer(buffer_distance1)
    river_gdf['buffered_2'] = river_gdf['geometry'].buffer(buffer_distance2)
    
    merged_list1 = [geom for geom in river_gdf['buffered_1']]
    merged_polygon = unary_union(merged_list1)
    merged_gdf_1 = gpd.GeoDataFrame(geometry=[merged_polygon], crs=river_gdf.crs)
    points_list_1 = []
    if merged_polygon and not merged_polygon.is_empty:
        for distance in range(0, int(merged_polygon.exterior.length), 10):  # 10 단위 간격으로 포인트 생성
            point = merged_polygon.exterior.interpolate(distance)
            points_list_1.append(Point(point))
    points_gdf_1 = gpd.GeoDataFrame(geometry=points_list_1, crs=river_gdf.crs)
    points_gdf_1.loc[:,"EL"]=sd_depth2
    
    merged_list2 = [geom for geom in river_gdf['buffered_2']]
    merged_polygon = unary_union(merged_list2)
    merged_gdf_2 = gpd.GeoDataFrame(geometry=[merged_polygon], crs=river_gdf.crs)
    points_list_2 = []
    if merged_polygon and not merged_polygon.is_empty:
        for distance in range(0, int(merged_polygon.exterior.length), 10):  # 10 단위 간격으로 포인트 생성
            point = merged_polygon.exterior.interpolate(distance)
            points_list_2.append(Point(point))
    points_gdf_2 = gpd.GeoDataFrame(geometry=points_list_2, crs=river_gdf.crs)
    points_gdf_2.loc[:,"EL"] = sd_depth
    
    all_EL_data = pd.concat([points_gdf_1,points_gdf_2])
    
    polygon = merged_gdf_2.geometry.iloc[0]
    gdf['inside_polygon'] = gdf.within(polygon)
    inside_points = gdf[gdf['inside_polygon']]
    outside_points = gdf[~gdf['inside_polygon']]
    
    points = np.array([(geom.x, geom.y) for geom in all_EL_data.geometry])
    values = all_EL_data['EL'].values
    linear_interpolator = LinearNDInterpolator(points, values)
    
    inside_points_coords = np.array([(geom.x, geom.y) for geom in inside_points.geometry])
    interpolated_values = linear_interpolator(inside_points_coords)
    inside_points['interpolated_EL'] = interpolated_values
    
    inside_points["lower EL"] = inside_points["upper EL"]-inside_points["interpolated_EL"]
    inside_points = inside_points.drop(columns = ['geometry','inside_polygon',"interpolated_EL"])
    outside_points = outside_points.drop(columns = ['geometry','inside_polygon'])
    sd_pd = pd.concat([inside_points, outside_points])
    sd_pd = sd_pd.reset_index(drop=True)

    return sd_pd

sd_pd = river_cell_linear(river_depth_area1, river_depth_area2, river_gdf, sd_pd)

sd_pd["X"] = (sd_pd["X"] - X_p) / cell_s + 1
sd_pd["Y"] = (Y_p - sd_pd["Y"]) / cell_s + 1
un_pd["X"] = (un_pd["X"] - X_p) / cell_s + 1
un_pd["Y"] = (Y_p - un_pd["Y"]) / cell_s + 1
sd_pd.rename(columns={'X': 'col', 'Y':'row'}, inplace=True)
un_pd.rename(columns={'X': 'col', 'Y':'row'}, inplace=True)

sd_pd = sd_pd.astype(float)
un_pd = un_pd.astype(float)

coords = sd_pd[['col', 'row']].to_numpy()
dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))

# 20m 이내의 지점 필터링 (자기 자신 제외)
within_20m_matrix = (dist_matrix <= np.sqrt(2)) & (dist_matrix > 0)

for i in range(sd_pd.shape[0]):
    within_20m_indices = np.where(within_20m_matrix[i])[0]
    row_value = sd_pd.loc[i, "row"]
    col_value = sd_pd.loc[i, "col"]
    if river_active == True:
        if ((river_pd["Row"] == row_value) & (river_pd["Column"] == col_value)).any():
            matching_index = river_pd[(river_pd["Row"] == row_value) & (river_pd["Column"] == col_value)].index
            if sd_pd.loc[i, "lower EL"] > river_pd.loc[matching_index[0] , "River_Bottom[m]"]:
                sd_pd.loc[i, "lower EL"] = river_pd.loc[matching_index[0] , "River_Bottom[m]"]
        
    
    if len(within_20m_indices) > 0:
        for j in within_20m_indices:
            cover_length = min(sd_pd.loc[i, "upper EL"], sd_pd.loc[j, "upper EL"]) - \
                           max(sd_pd.loc[i, "lower EL"], sd_pd.loc[j, "lower EL"])
            if cover_length < admit_range and sd_pd.loc[i, "upper EL"] > sd_pd.loc[j, "upper EL"]:
                re_el = admit_range + cover_length
                new_lower = -(admit_range - sd_pd.loc[j, "upper EL"])
                sd_pd.loc[i, "lower EL"] = new_lower
    
    # 진행 상황 출력
    if i % 100 == 0:  # 매 100번째 루프마다 진행 상황 출력
        print(f"Progress: {i/sd_pd.shape[0]*100:.2f}%")
                              

# 불필요한 열 제거
sd_pd.drop(columns=["upper EL"], inplace=True)
sd_pd.rename(columns={'lower EL': 'EL'}, inplace=True)
sd_pd['퇴적']=0

# 데이터프레임의 모든 열을 float로 변환

while un_pd.shape[0] > 0:  # un_pd의 행 개수가 0이 될 때까지 반복
    ok_data = []
    pending_data = []  # 다음 루프로 넘길 데이터를 저장할 리스트

    for i in range(un_pd.shape[0]):
        # 거리 계산
        distances = np.sqrt((sd_pd["row"].values - un_pd.iloc[i]["row"])**2 + 
                            (sd_pd["col"].values - un_pd.iloc[i]["col"])**2)

        within_range_indices = np.where((distances <= effect_cell_count*math.sqrt(2)) & (distances > 0))[0]
        
        if len(within_range_indices) > 0:
            mean_el = np.average(sd_pd.iloc[within_range_indices]["EL"].values)
            std_el = np.std(sd_pd.iloc[within_range_indices]["EL"].values)
            
            if std_el > 5:
                ratio = mean_el + 1.5 * std_el
            elif 3 < std_el <= 5:
                ratio = mean_el + 2 * std_el
            elif 1 < std_el <= 3:
                ratio = mean_el + 3 * std_el
            else:
                ratio = mean_el + 4 * std_el
            
            if ratio > un_pd.iloc[i, un_pd.columns.get_loc("EL")] - 5:
                ratio = un_pd.iloc[i, un_pd.columns.get_loc("EL")] - 5
        
            row_value = un_pd.loc[i, "row"]
            col_value = un_pd.loc[i, "col"]
            if river_active == True:    
                if ((river_pd["Row"] == row_value) & (river_pd["Column"] == col_value)).any():
                    matching_index = river_pd[(river_pd["Row"] == row_value) & (river_pd["Column"] == col_value)].index
                    if ratio > river_pd.loc[matching_index[0] , "River_Bottom[m]"]:
                        ratio = river_pd.loc[matching_index[0] , "River_Bottom[m]"]
            
            
            un_pd.loc[i,"퇴적"]=un_pd.loc[i,"EL"]
            un_pd.iat[i, un_pd.columns.get_loc("EL")] = ratio
            
            ok_data.append(un_pd.iloc[i])
        else:
            # 50m 이내에 점이 없으면 다음 루프로 넘김
            pending_data.append(un_pd.iloc[i])
        
        if i % 10 == 0:
            print(f"Processing {i/un_pd.shape[0]*100:.2f}% done")

    ok_data_df = pd.DataFrame(ok_data)
    
    # sd_pd에 ok_data_df 추가
    sd_pd = pd.concat([sd_pd, ok_data_df], ignore_index=True)
    
    # un_pd에 남은 데이터를 다시 설정 (다음 루프로 넘길 데이터)
    un_pd = pd.DataFrame(pending_data).reset_index(drop=True)
    
    # 루프 종료 조건: un_pd에 더 이상 데이터가 없으면 종료
    if un_pd.shape[0] == 0:
        break

for i in range(iterative):
    for cell in range(sd_pd.shape[0]):
        if sd_pd.loc[cell,"퇴적"] != 0:
            distances = np.sqrt((sd_pd["row"].values - sd_pd.iloc[cell]["row"])**2 + 
                                (sd_pd["col"].values - sd_pd.iloc[cell]["col"])**2)

            within_range_indices = np.where((distances <= math.sqrt(2)) & (distances > 0))[0]
            
            mean_el = np.average(sd_pd.iloc[within_range_indices]["EL"].values)
            std_el = np.std(sd_pd.iloc[within_range_indices]["EL"].values)
            
            if version == 0:
                ratio=mean_el
            elif version ==1 :
                if std_el > 5:
                    ratio = mean_el + 1.5 * std_el
                elif 3 < std_el <= 5:
                    ratio = mean_el + 2 * std_el
                elif 1 < std_el <= 3:
                    ratio = mean_el + 3 * std_el
                else:
                    ratio = mean_el + 4 * std_el
            
            if ratio > sd_pd.iloc[cell, sd_pd.columns.get_loc("퇴적")] - 5:
                ratio = sd_pd.iloc[cell, sd_pd.columns.get_loc("퇴적")] - 5
                
            row_value = sd_pd.loc[i, "row"]
            col_value = sd_pd.loc[i, "col"]        
            if river_active == True:
                if ((river_pd["Row"] == row_value) & (river_pd["Column"] == col_value)).any():
                    matching_index = river_pd[(river_pd["Row"] == row_value) & (river_pd["Column"] == col_value)].index
                    if ratio > river_pd.loc[matching_index[0] , "River_Bottom[m]"]:
                        ratio = river_pd.loc[matching_index[0] , "River_Bottom[m]"]
            
            sd_pd.loc[cell,"EL"] = ratio

            if cell % 10 == 0:
                print(f"Iterative {(cell+i*sd_pd.shape[0])/(sd_pd.shape[0]*iterative)*100:.2f}% done")

sd_pd["Layer"] = 1
sd_pd.drop(columns=["퇴적"], inplace=True)

# 데이터프레임을 CSV 파일로 저장
out = os.path.join(file_paths, "output.txt")
sd_pd.to_csv(out, index=False)


