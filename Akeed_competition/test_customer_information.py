import pandas as pd 
import numpy as np
from datetime import datetime,date,time
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
from utils import *
print('test customer information creation started (location +customer information')
test_locations_path       = os.path.join(os.getcwd(),'raw_input_data','test_locations.csv')
test_location_df          = pd.read_csv(test_locations_path)


test_customer_path        = os.path.join(os.getcwd(),'raw_input_data','test_customers.csv')
test_customer_df          = pd.read_csv(test_customer_path)
test_customer_df          = test_customer_df.rename(columns = {'akeed_customer_id':'customer_id',
                                                                   'status':'customer_status',
                                                                   'verified':'customer_verified'
                                                                  })

test_location_df          = test_location_df.rename(columns={'latitude':'cust_latitude',
                                                                 'longitude':'cust_longitude'
                                                                })

test_customer_df.drop_duplicates(subset='customer_id',inplace=True)


test_customer_information = join_customer_location(test_customer_df,test_location_df)


test_customer_information['customer_manhattan_distance']   = manhattan_distance(test_customer_information['cust_latitude'].values,
                                                                                 test_customer_information['cust_longitude'].values
                                                                                )

test_customer_information['customer_Bearing_distance']     = bearing_array(test_customer_information['cust_latitude'].values,
                                                                                  test_customer_information['cust_longitude'].values
                                                                              )

test_customer_information['customer_bearing_by_manhattan'] = (test_customer_information['customer_Bearing_distance']/test_customer_information['customer_manhattan_distance'])

reduce_mem_usage(test_customer_information)

count_df                   = my_groupby(test_customer_information,['customer_id'],{'location_number':'count'},{'location_number':'customer_location_count'})

test_customer_information = data_left_join(test_customer_information,count_df,['customer_id'])

test_customer_mean_lat    = my_groupby(test_customer_information,['customer_id'],
                                        {'cust_latitude':'mean'},{'cust_latitude':'cust_latitude_mean'})

test_customer_mean_long   = my_groupby(test_customer_information,['customer_id'],
                                        {'cust_longitude':'mean'},{'cust_longitude':'cust_longitude_mean'})

test_customer_mean_bearing= my_groupby(test_customer_information,['customer_id'],
                                        {'customer_Bearing_distance':'mean'},{'customer_Bearing_distance':'cust_bearing_mean'})


test_customer_std_lat     = my_groupby(test_customer_information,['customer_id'],
                                        {'cust_latitude':'std'},{'cust_latitude':'cust_latitude_std'})

test_customer_std_long    = my_groupby(test_customer_information,['customer_id'],
                                        {'cust_longitude':'std'},{'cust_longitude':'cust_longitude_std'})

test_customer_std_bearing = my_groupby(test_customer_information,['customer_id'],
                                        {'customer_Bearing_distance':'std'},{'customer_Bearing_distance':'cust_bearing_std'})

reduce_mem_usage(test_customer_information)
test_customer_information                         = data_left_join(test_customer_information,test_customer_mean_lat,['customer_id'])
test_customer_information                         = data_left_join(test_customer_information,test_customer_mean_long,['customer_id'])
test_customer_information                         = data_left_join(test_customer_information,test_customer_mean_bearing,['customer_id'])


test_customer_information                         = data_left_join(test_customer_information,test_customer_std_lat,['customer_id'])
test_customer_information                         = data_left_join(test_customer_information,test_customer_std_long,['customer_id'])
test_customer_information                         = data_left_join(test_customer_information,test_customer_std_bearing,['customer_id'])
reduce_mem_usage(test_customer_information)

test_customer_information['cust_latitude_std']    = test_customer_information['cust_latitude_std'].replace({np.nan:0.0005})
test_customer_information['cust_longitude_std']   = test_customer_information['cust_longitude_std'].replace({np.nan:0.005})
test_customer_information['cust_bearing_std']     = test_customer_information['cust_bearing_std'].replace({np.nan:0.0005})

test_customer_information['lat_by_long']          = test_customer_information['cust_latitude']/test_customer_information['cust_longitude']

test_customer_std_lat_by_long                     = my_groupby(test_customer_information,['customer_id'],
                                                         {'lat_by_long':'std'},{'lat_by_long':'cust_lat_by_long_std'})
test_customer_information                         = data_left_join(test_customer_information,test_customer_std_lat_by_long,['customer_id'])

test_customer_mean_lat_by_long                    = my_groupby(test_customer_information,['customer_id'],
                                                         {'lat_by_long':'mean'},{'lat_by_long':'cust_lat_by_long_mean'})
test_customer_information                         = data_left_join(test_customer_information,test_customer_mean_lat_by_long,['customer_id'])

test_customer_information['cust_lat_by_long_std'] = test_customer_information['cust_lat_by_long_std'].replace({np.nan:0.0005})

test_customer_information['normalised_lat']       = (test_customer_information['cust_latitude']-test_customer_information['cust_latitude_mean'])/test_customer_information['cust_latitude_std']
test_customer_information['normalised_long']      = (test_customer_information['cust_longitude']-test_customer_information['cust_longitude_mean'])/test_customer_information['cust_longitude_std']

reduce_mem_usage(test_customer_information)
test_customer_information['north_south']          = test_customer_information['cust_latitude'].apply(location_north_south)
test_customer_information['east_west']            = test_customer_information['cust_longitude'].apply(location_east_west)

reduce_mem_usage(test_customer_information)


test_customer_information['cust_latitude_cv']=test_customer_information['cust_latitude_std']/test_customer_information['cust_latitude_mean']
test_customer_information['cust_longitude_cv']=test_customer_information['cust_longitude_std']/test_customer_information['cust_longitude_mean']
test_customer_information['cust_bearing_cv']=test_customer_information['cust_bearing_std']/test_customer_information['cust_bearing_mean']
test_customer_information['cust_lat_by_long_cv']=test_customer_information['cust_lat_by_long_std']/test_customer_information['cust_lat_by_long_mean']
reduce_mem_usage(test_customer_information)

print('test customer information creation ends (location +customer information')