import pandas as pd 
import numpy as np
from datetime import datetime,date,time
pd.set_option('display.max_columns', None)
import warnings
import os 
warnings.filterwarnings("ignore")
from utils import *

print('Train customer creation started (location+customer information')
################## Step 1:- Location and customer information loaded ################################

train_locations_path       = os.path.join(os.getcwd(),'raw_input_data','train_locations.csv')
train_location_df          = pd.read_csv(train_locations_path)


train_customer_path        = os.path.join(os.getcwd(),'raw_input_data','train_customers.csv')
train_customer_df          = pd.read_csv(train_customer_path)

################ We change the column name to make sure it is not confusing with the vendor column names 
################ when we do a join later in the codes.

train_customer_df          = train_customer_df.rename(columns = {'akeed_customer_id':'customer_id',
                                                                   'status':'customer_status',
                                                                   'verified':'customer_verified'
                                                                  })

train_location_df          = train_location_df.rename(columns={'latitude':'cust_latitude',
                                                                 'longitude':'cust_longitude'
                                                                })

train_customer_df.drop_duplicates(subset='customer_id',inplace=True)

################### Step 2:- Joining the dataset for location and customers #########################

train_customer_information = join_customer_location(train_customer_df,train_location_df)


################## Step 3:- COmputation of distance measures ########################################

train_customer_information['customer_manhattan_distance']   = manhattan_distance(train_customer_information['cust_latitude'].values,
                                                                                 train_customer_information['cust_longitude'].values
                                                                                )

train_customer_information['customer_Bearing_distance']     = bearing_array(train_customer_information['cust_latitude'].values,
                                                                                  train_customer_information['cust_longitude'].values
                                                                              )

train_customer_information['customer_bearing_by_manhattan'] = (train_customer_information['customer_Bearing_distance']/train_customer_information['customer_manhattan_distance'])

reduce_mem_usage(train_customer_information)


################### Step 4:- Computation of aggregate measures ######################################

count_df                   = my_groupby(train_customer_information,['customer_id'],{'location_number':'count'},{'location_number':'customer_location_count'})

train_customer_information = data_left_join(train_customer_information,count_df,['customer_id'])

train_customer_mean_lat    = my_groupby(train_customer_information,['customer_id'],
                                        {'cust_latitude':'mean'},{'cust_latitude':'cust_latitude_mean'})

train_customer_mean_long   = my_groupby(train_customer_information,['customer_id'],
                                        {'cust_longitude':'mean'},{'cust_longitude':'cust_longitude_mean'})

train_customer_mean_bearing= my_groupby(train_customer_information,['customer_id'],
                                        {'customer_Bearing_distance':'mean'},{'customer_Bearing_distance':'cust_bearing_mean'})


train_customer_std_lat     = my_groupby(train_customer_information,['customer_id'],
                                        {'cust_latitude':'std'},{'cust_latitude':'cust_latitude_std'})

train_customer_std_long    = my_groupby(train_customer_information,['customer_id'],
                                        {'cust_longitude':'std'},{'cust_longitude':'cust_longitude_std'})

train_customer_std_bearing = my_groupby(train_customer_information,['customer_id'],
                                        {'customer_Bearing_distance':'std'},{'customer_Bearing_distance':'cust_bearing_std'})

reduce_mem_usage(train_customer_information)

train_customer_information                         = data_left_join(train_customer_information,train_customer_mean_lat,['customer_id'])
train_customer_information                         = data_left_join(train_customer_information,train_customer_mean_long,['customer_id'])
train_customer_information                         = data_left_join(train_customer_information,train_customer_mean_bearing,['customer_id'])

reduce_mem_usage(train_customer_information)
train_customer_information                         = data_left_join(train_customer_information,train_customer_std_lat,['customer_id'])
train_customer_information                         = data_left_join(train_customer_information,train_customer_std_long,['customer_id'])
train_customer_information                         = data_left_join(train_customer_information,train_customer_std_bearing,['customer_id'])


train_customer_information['cust_latitude_std']    = train_customer_information['cust_latitude_std'].replace({np.nan:0.0005})
train_customer_information['cust_longitude_std']   = train_customer_information['cust_longitude_std'].replace({np.nan:0.005})
train_customer_information['cust_bearing_std']     = train_customer_information['cust_bearing_std'].replace({np.nan:0.0005})

train_customer_information['lat_by_long']          = train_customer_information['cust_latitude']/train_customer_information['cust_longitude']

train_customer_std_lat_by_long                     = my_groupby(train_customer_information,['customer_id'],
                                                         {'lat_by_long':'std'},{'lat_by_long':'cust_lat_by_long_std'})
train_customer_information                         = data_left_join(train_customer_information,train_customer_std_lat_by_long,['customer_id'])

train_customer_mean_lat_by_long                    = my_groupby(train_customer_information,['customer_id'],
                                                         {'lat_by_long':'mean'},{'lat_by_long':'cust_lat_by_long_mean'})
train_customer_information                         = data_left_join(train_customer_information,train_customer_mean_lat_by_long,['customer_id'])

reduce_mem_usage(train_customer_information)


train_customer_information['cust_lat_by_long_std'] = train_customer_information['cust_lat_by_long_std'].replace({np.nan:0.0005})

train_customer_information['normalised_lat']       = (train_customer_information['cust_latitude']-train_customer_information['cust_latitude_mean'])/train_customer_information['cust_latitude_std']
train_customer_information['normalised_long']      = (train_customer_information['cust_longitude']-train_customer_information['cust_longitude_mean'])/train_customer_information['cust_longitude_std']



######################## Step 5:- Computation of distance measures of customers based on latitude and longitude #############
train_customer_information['north_south']          = train_customer_information['cust_latitude'].apply(location_north_south)
train_customer_information['east_west']            = train_customer_information['cust_longitude'].apply(location_east_west)


reduce_mem_usage(train_customer_information)


train_customer_information['cust_latitude_cv']=train_customer_information['cust_latitude_std']/train_customer_information['cust_latitude_mean']
train_customer_information['cust_longitude_cv']=train_customer_information['cust_longitude_std']/train_customer_information['cust_longitude_mean']
train_customer_information['cust_bearing_cv']=train_customer_information['cust_bearing_std']/train_customer_information['cust_bearing_mean']
train_customer_information['cust_lat_by_long_cv']=train_customer_information['cust_lat_by_long_std']/train_customer_information['cust_lat_by_long_mean']

reduce_mem_usage(train_customer_information)
#train_customer_information.head()

print('Train customer creation ends (location+customer information')