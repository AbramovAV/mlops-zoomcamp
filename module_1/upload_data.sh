wget https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-01.parquet
wget https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet
mkdir -p data
mv fhv_tripdata_2021-01.parquet data/fhv_tripdata_2021-01.parquet
mv fhv_tripdata_2021-02.parquet data/fhv_tripdata_2021-02.parquet