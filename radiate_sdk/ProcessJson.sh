for folder in 'junction_1_13' 'motorway_1_0' 'junction_1_2' 'junction_1_0' 'city_2_0' 'city_4_0' 'junction_1_7' 'junction_1_6' 'junction_2_3' 'city_1_1' 'junction_2_1' 'city_3_2' 'junction_3_0' 'motorway_2_1' 'city_3_1' 'junction_3_1' 'city_3_0' 'city_1_0' 'rural_1_1' 'city_3_3' 'junction_2_2' 'city_5_0' 'motorway_2_0' 'junction_1_3' 'rural_1_3' 'junction_1_4' 'junction_1_1' 'junction_1_14' 'junction_1_5' 'junction_1_15' 'junction_2_0' 'junction_2_5' 'junction_1_9' 'junction_3_3' 'fog_8_2' 'rain_2_0' 'city_6_0' 'fog_8_0' 'city_1_3' 'junction_1_8' 'night_1_1' 'night_1_3' 'rain_4_1' 'night_1_0' 'junction_1_10' 'city_7_0' 'junction_2_6' 'fog_6_0' 'rain_3_0' 'rain_4_0' 'night_1_5' 'night_1_4' 'motorway_2_2' 'junction_1_11' 'night_1_2' 'snow_1_0' 'city_3_7' 'junction_3_2' 'fog_8_1' 'junction_1_12'
do
    mkdir /data/RADIATE/${folder}/labels
    python3 ProcessJson.py ${folder}
done

mkdir /data/Radiate_Good/labels
mkdir /data/Radiate_Good_and_Bad/labels
mkdir /data/Radiate_Test/labels

# train_good_weather
for folder in 'junction_1_2' 'junction_1_0' 'city_2_0' 'city_4_0' 'junction_1_7' 'junction_1_6' 'junction_2_3' 'city_1_1' 'junction_2_1' 'city_3_2' 'junction_3_0' 'motorway_2_1' 'city_3_1' 'junction_3_1' 'city_3_0' 'city_1_0' 'rural_1_1' 'city_3_3' 'junction_2_2' 'city_5_0' 'motorway_2_0' 'junction_1_3' 'rural_1_3' 'junction_1_4' 'junction_1_1' 'junction_1_14' 'junction_1_5' 'junction_1_15' 'junction_2_0'
do
    cp /data/RADIATE/${folder}/labels/* /data/Radiate_Good/labels
    printf "Finish, %s\n" ${folder}
done
# train_good_and_bad_weather
for folder in 'junction_3_3' 'fog_8_2' 'rain_2_0' 'city_6_0' 'fog_8_0' 'city_1_3' 'junction_1_8' 'night_1_1' 'night_1_3' 'rain_4_1'
do
  cp /data/RADIATE/${folder}/labels/* /data/Radiate_Good_and_Bad/labels
  printf "Finish, %s\n" ${folder}
done
# test
for folder in 'city_7_0' 'junction_2_6' 'fog_6_0' 'rain_3_0' 'rain_4_0' 'night_1_5' 'night_1_4' 'motorway_2_2' 'junction_1_11' 'night_1_2' 'snow_1_0' 'city_3_7' 'junction_3_2' 'fog_8_1' 'junction_1_12'
do
  cp /data/RADIATE/${folder}/labels/* /data/Radiate_Test/labels
  printf "Finish, %s\n" ${folder}
done