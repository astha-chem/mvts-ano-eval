
# smap_msl
# download dataset
mkdir data/raw/smap_msl
curl --output data/raw/smap_msl/data.zip https://s3-us-west-2.amazonaws.com/telemanom/data.zip
# unzip the files, rename and cleanup
unzip data/raw/smap_msl/data.zip -d data/raw/smap_msl/ && mv data/raw/smap_msl/data/* data/raw/smap_msl/. && rm -r data/raw/smap_msl/data  && rm -r data/raw/smap_msl/data.zip
# download anomaly information
curl --output data/raw/smap_msl/labeled_anomalies.csv https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv

# damadics
mkdir data/raw/damadics/
mkdir data/raw/damadics/raw
cd data/raw/damadics/raw
curl https://iair.mchtr.pw.edu.pl/content/download/163/817/file/Lublin_all_data_part1.zip --output part1.zip
curl https://iair.mchtr.pw.edu.pl/content/download/164/821/file/Lublin_all_data_part2.zip --output part2.zip
curl https://iair.mchtr.pw.edu.pl/content/download/165/825/file/Lublin_all_data_part3.zip --output part3.zip
curl https://iair.mchtr.pw.edu.pl/content/download/166/829/file/Lublin_all_data_part4.zip --output part4.zip
unzip part1.zip && unzip part2.zip && unzip part3.zip && unzip part4.zip
mv Lublin_all_data/* .
rm part1.zip && rm part2.zip && rm part3.zip && rm part4.zip && rm -r Lublin_all_data
cd ../../../..

# SMD
mkdir data/raw/smd && cd data/raw/smd
curl -L https://github.com/NetManAIOps/OmniAnomaly/tarball/master | tar xz --strip-components=1
mv ServerMachineDataset/ ../ServerMachineDataset
cd ../../..
rm -r data/raw/smd

# SKAB
mkdir data/raw/skab && cd data/raw/skab
curl -L https://github.com/waico/SKAB/tarball/master | tar xz --strip-components=1
rm -r docs
rm -r notebooks
rm -r utils
rm -v !("data") && mv data/* ../skab
rm -r data/

