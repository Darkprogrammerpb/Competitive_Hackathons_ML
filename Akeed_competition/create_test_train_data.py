from train_customer_information import *
from test_customer_information import *
from customer_vendor_order_data import *

import warnings
warnings.filterwarnings("ignore")

print('Basic train and test data without aggregations creation started')
############### Create train data ##################
vendor_df['key']                                   = 0
train_customer_information['key']                  = 0

train_vendor_data_merged                           = pd.merge(train_customer_information,vendor_df,how='outer',on='key')
reduce_mem_usage(train_vendor_data_merged)
train_vendor_data_merged                           = train_vendor_data_merged.drop('key',axis=1)
reduce_mem_usage(train_vendor_data_merged)


train_vendor_data_merged['dist_km']                 = haversine_distance(train_vendor_data_merged,'vendor_latitude', 'vendor_longitude', 'cust_latitude', 'cust_longitude')

train_vendor_data_merged['midpoint_lat']            = (train_vendor_data_merged['vendor_latitude']+train_vendor_data_merged['cust_latitude'])/2


train_vendor_data_merged['midpoint_long']           = (train_vendor_data_merged['vendor_longitude']+train_vendor_data_merged['cust_longitude'])/2

train_vendor_data_merged['id']                      = train_vendor_data_merged['id'].astype(int)
train_vendor_data_merged['location_number']         = train_vendor_data_merged['location_number'].astype(int)
train_vendor_data_merged['CID X LOC_NUM X VENDOR']  = train_vendor_data_merged['customer_id']+' X '+train_vendor_data_merged['location_number'].astype(str)+' X '+train_vendor_data_merged['id'].astype(str)
reduce_mem_usage(train_vendor_data_merged)

combinations_served                = order_df['CID X LOC_NUM X VENDOR'].values.tolist()
train_vendor_data_merged['target'] = train_vendor_data_merged['CID X LOC_NUM X VENDOR'].isin(combinations_served).astype(int).values
reduce_mem_usage(train_vendor_data_merged)

################# Create test data ######################

vendor_df['key']                                    = 0
test_customer_information['key']                    = 0

test_vendor_data_merged                             = pd.merge(test_customer_information,vendor_df,how='outer',on='key')
reduce_mem_usage(test_vendor_data_merged)
test_vendor_data_merged                             = test_vendor_data_merged.drop('key',axis=1)

test_vendor_data_merged['dist_km']                = haversine_distance(test_vendor_data_merged,'vendor_latitude', 'vendor_longitude', 'cust_latitude', 'cust_longitude')

test_vendor_data_merged['midpoint_lat']            = (test_vendor_data_merged['vendor_latitude']+test_vendor_data_merged['cust_latitude'])/2


test_vendor_data_merged['midpoint_long']           = (test_vendor_data_merged['vendor_longitude']+test_vendor_data_merged['cust_longitude'])/2


test_vendor_data_merged['id']                     = test_vendor_data_merged['id'].astype(int)
test_vendor_data_merged['location_number']        = test_vendor_data_merged['location_number'].astype(int)
test_vendor_data_merged['CID X LOC_NUM X VENDOR'] = test_vendor_data_merged['customer_id']+' X '+test_vendor_data_merged['location_number'].astype(str)+' X '+test_vendor_data_merged['id'].astype(str)
reduce_mem_usage(test_vendor_data_merged)

print('Basic train and test data without aggregations creation ended!')
############## Additional date columns to be computed 

print('Additional date columns for vendor and customer started!')
train_vendor_data_merged['vendor_created_at'] = pd.to_datetime(train_vendor_data_merged['vendor_created_at'])
train_vendor_data_merged['vendor_updated_at'] = pd.to_datetime(train_vendor_data_merged['vendor_updated_at'])
train_vendor_data_merged['created_at'] = pd.to_datetime(train_vendor_data_merged['created_at'])
train_vendor_data_merged['updated_at'] = pd.to_datetime(train_vendor_data_merged['updated_at'])
train_vendor_data_merged =  reduce_mem_usage(train_vendor_data_merged)

test_vendor_data_merged['vendor_created_at'] = pd.to_datetime(test_vendor_data_merged['vendor_created_at'])
test_vendor_data_merged['vendor_updated_at'] = pd.to_datetime(test_vendor_data_merged['vendor_updated_at'])
test_vendor_data_merged['created_at'] = pd.to_datetime(test_vendor_data_merged['created_at'])
test_vendor_data_merged['updated_at'] = pd.to_datetime(test_vendor_data_merged['updated_at'])
test_vendor_data_merged =  reduce_mem_usage(test_vendor_data_merged)


train_vendor_data_merged['vendor_created_customer_created_diff']=np.abs((train_vendor_data_merged['vendor_created_at']-train_vendor_data_merged['created_at']).dt.days)
test_vendor_data_merged['vendor_created_customer_created_diff']=np.abs((test_vendor_data_merged['vendor_created_at']-test_vendor_data_merged['created_at']).dt.days)

train_vendor_data_merged['vendor_created_customer_updated_diff']=np.abs((train_vendor_data_merged['vendor_created_at']-train_vendor_data_merged['updated_at']).dt.days)
test_vendor_data_merged['vendor_created_customer_updated_diff']=np.abs((test_vendor_data_merged['vendor_created_at']-test_vendor_data_merged['updated_at']).dt.days)


train_vendor_data_merged['vendor_updated_customer_created_diff']=np.abs((train_vendor_data_merged['vendor_updated_at']-train_vendor_data_merged['created_at']).dt.days)
test_vendor_data_merged['vendor_updated_customer_created_diff']=np.abs((test_vendor_data_merged['vendor_updated_at']-test_vendor_data_merged['created_at']).dt.days)

train_vendor_data_merged['vendor_updated_customer_updated_diff']=np.abs((train_vendor_data_merged['vendor_updated_at']-train_vendor_data_merged['updated_at']).dt.days)
test_vendor_data_merged['vendor_updated_customer_updated_diff']=np.abs((test_vendor_data_merged['vendor_updated_at']-test_vendor_data_merged['updated_at']).dt.days)

reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)
 
print('Redundant columns to be dropped ')
drop_cols=['customer_status','customer_verified','language','vendor_tag_name','vendor_updated_at','vendor_created_at','created_at','updated_at']

train_vendor_data_merged=train_vendor_data_merged.drop(drop_cols,axis=1)
test_vendor_data_merged=test_vendor_data_merged.drop(drop_cols,axis=1)

reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)


train_vendor_data_merged['can_be_reached'] = np.log1p(train_vendor_data_merged['dist_km'])<train_vendor_data_merged['serving_distance']
train_vendor_data_merged['can_be_reached'] = train_vendor_data_merged['can_be_reached'].astype(int)


test_vendor_data_merged['can_be_reached'] = np.log1p(test_vendor_data_merged['dist_km'])<test_vendor_data_merged['serving_distance']
test_vendor_data_merged['can_be_reached'] = test_vendor_data_merged['can_be_reached'].astype(int)



reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)

print('Rotational parameters for latitudes and longitudes started!')
###################### Rotated lat and long ####################
train_vendor_data_merged['customer_long_15'] = train_vendor_data_merged['cust_longitude']*np.cos(15* np.pi / 180) - train_vendor_data_merged['cust_latitude']*np.sin(15* np.pi/180)
train_vendor_data_merged['customer_long_30'] = train_vendor_data_merged['cust_longitude']*np.cos(30* np.pi / 180) - train_vendor_data_merged['cust_latitude']*np.sin(30* np.pi/180)
train_vendor_data_merged['customer_long_45'] = train_vendor_data_merged['cust_longitude']*np.cos(45* np.pi / 180) - train_vendor_data_merged['cust_latitude']*np.sin(45* np.pi/180)
train_vendor_data_merged['customer_long_60'] = train_vendor_data_merged['cust_longitude']*np.cos(60* np.pi / 180) - train_vendor_data_merged['cust_latitude']*np.sin(60* np.pi/180)
train_vendor_data_merged['customer_long_75'] = train_vendor_data_merged['cust_longitude']*np.cos(75* np.pi / 180) - train_vendor_data_merged['cust_latitude']*np.sin(75* np.pi/180)
reduce_mem_usage(train_vendor_data_merged)

train_vendor_data_merged['customer_lat_15'] = train_vendor_data_merged['cust_longitude']*np.sin(15* np.pi / 180) + train_vendor_data_merged['cust_latitude']*np.cos(15* np.pi/180)
train_vendor_data_merged['customer_lat_30'] = train_vendor_data_merged['cust_longitude']*np.sin(30* np.pi / 180) + train_vendor_data_merged['cust_latitude']*np.cos(30* np.pi/180)
train_vendor_data_merged['customer_lat_45'] = train_vendor_data_merged['cust_longitude']*np.sin(45* np.pi / 180) + train_vendor_data_merged['cust_latitude']*np.cos(45* np.pi/180)
train_vendor_data_merged['customer_lat_60'] = train_vendor_data_merged['cust_longitude']*np.sin(60* np.pi / 180) + train_vendor_data_merged['cust_latitude']*np.cos(60* np.pi/180)
train_vendor_data_merged['customer_lat_75'] = train_vendor_data_merged['cust_longitude']*np.sin(75* np.pi / 180) + train_vendor_data_merged['cust_latitude']*np.cos(75* np.pi/180)
reduce_mem_usage(train_vendor_data_merged)

train_vendor_data_merged['vendor_long_15'] = train_vendor_data_merged['vendor_longitude']*np.cos(15* np.pi / 180) - train_vendor_data_merged['vendor_latitude']*np.sin(15* np.pi/180)
train_vendor_data_merged['vendor_long_30'] = train_vendor_data_merged['vendor_longitude']*np.cos(30* np.pi / 180) - train_vendor_data_merged['vendor_latitude']*np.sin(30* np.pi/180)
train_vendor_data_merged['vendor_long_45'] = train_vendor_data_merged['vendor_longitude']*np.cos(45* np.pi / 180) - train_vendor_data_merged['vendor_latitude']*np.sin(45* np.pi/180)
train_vendor_data_merged['vendor_long_60'] = train_vendor_data_merged['vendor_longitude']*np.cos(60* np.pi / 180) - train_vendor_data_merged['vendor_latitude']*np.sin(60* np.pi/180)
train_vendor_data_merged['vendor_long_75'] = train_vendor_data_merged['vendor_longitude']*np.cos(75* np.pi / 180) - train_vendor_data_merged['vendor_latitude']*np.sin(75* np.pi/180)
reduce_mem_usage(train_vendor_data_merged)

train_vendor_data_merged['vendor_lat_15'] = train_vendor_data_merged['vendor_longitude']*np.sin(15* np.pi / 180) + train_vendor_data_merged['vendor_latitude']*np.cos(15* np.pi/180)
train_vendor_data_merged['vendor_lat_30'] = train_vendor_data_merged['vendor_longitude']*np.sin(30* np.pi / 180) + train_vendor_data_merged['vendor_latitude']*np.cos(30* np.pi/180)
train_vendor_data_merged['vendor_lat_45'] = train_vendor_data_merged['vendor_longitude']*np.sin(45* np.pi / 180) + train_vendor_data_merged['vendor_latitude']*np.cos(45* np.pi/180)
train_vendor_data_merged['vendor_lat_60'] = train_vendor_data_merged['vendor_longitude']*np.sin(60* np.pi / 180) + train_vendor_data_merged['vendor_latitude']*np.cos(60* np.pi/180)
train_vendor_data_merged['vendor_lat_75'] = train_vendor_data_merged['vendor_longitude']*np.sin(75* np.pi / 180) + train_vendor_data_merged['vendor_latitude']*np.cos(75* np.pi/180)
reduce_mem_usage(train_vendor_data_merged)


test_vendor_data_merged['customer_long_15'] = test_vendor_data_merged['cust_longitude']*np.cos(15* np.pi / 180) - test_vendor_data_merged['cust_latitude']*np.sin(15* np.pi/180)
test_vendor_data_merged['customer_long_30'] = test_vendor_data_merged['cust_longitude']*np.cos(30* np.pi / 180) - test_vendor_data_merged['cust_latitude']*np.sin(30* np.pi/180)
test_vendor_data_merged['customer_long_45'] = test_vendor_data_merged['cust_longitude']*np.cos(45* np.pi / 180) - test_vendor_data_merged['cust_latitude']*np.sin(45* np.pi/180)
test_vendor_data_merged['customer_long_60'] = test_vendor_data_merged['cust_longitude']*np.cos(60* np.pi / 180) - test_vendor_data_merged['cust_latitude']*np.sin(60* np.pi/180)
test_vendor_data_merged['customer_long_75'] = test_vendor_data_merged['cust_longitude']*np.cos(75* np.pi / 180) - test_vendor_data_merged['cust_latitude']*np.sin(75* np.pi/180)
reduce_mem_usage(test_vendor_data_merged)

test_vendor_data_merged['customer_lat_15'] = test_vendor_data_merged['cust_longitude']*np.sin(15* np.pi / 180) + test_vendor_data_merged['cust_latitude']*np.cos(15* np.pi/180)
test_vendor_data_merged['customer_lat_30'] = test_vendor_data_merged['cust_longitude']*np.sin(30* np.pi / 180) + test_vendor_data_merged['cust_latitude']*np.cos(30* np.pi/180)
test_vendor_data_merged['customer_lat_45'] = test_vendor_data_merged['cust_longitude']*np.sin(45* np.pi / 180) + test_vendor_data_merged['cust_latitude']*np.cos(45* np.pi/180)
test_vendor_data_merged['customer_lat_60'] = test_vendor_data_merged['cust_longitude']*np.sin(60* np.pi / 180) + test_vendor_data_merged['cust_latitude']*np.cos(60* np.pi/180)
test_vendor_data_merged['customer_lat_75'] = test_vendor_data_merged['cust_longitude']*np.sin(75* np.pi / 180) + test_vendor_data_merged['cust_latitude']*np.cos(75* np.pi/180)
reduce_mem_usage(test_vendor_data_merged)

test_vendor_data_merged['vendor_long_15'] = test_vendor_data_merged['vendor_longitude']*np.cos(15* np.pi / 180) - test_vendor_data_merged['vendor_latitude']*np.sin(15* np.pi/180)
test_vendor_data_merged['vendor_long_30'] = test_vendor_data_merged['vendor_longitude']*np.cos(30* np.pi / 180) - test_vendor_data_merged['vendor_latitude']*np.sin(30* np.pi/180)
test_vendor_data_merged['vendor_long_45'] = test_vendor_data_merged['vendor_longitude']*np.cos(45* np.pi / 180) - test_vendor_data_merged['vendor_latitude']*np.sin(45* np.pi/180)
test_vendor_data_merged['vendor_long_60'] = test_vendor_data_merged['vendor_longitude']*np.cos(60* np.pi / 180) - test_vendor_data_merged['vendor_latitude']*np.sin(60* np.pi/180)
test_vendor_data_merged['vendor_long_75'] = test_vendor_data_merged['vendor_longitude']*np.cos(75* np.pi / 180) - test_vendor_data_merged['vendor_latitude']*np.sin(75* np.pi/180)
reduce_mem_usage(test_vendor_data_merged)
test_vendor_data_merged['vendor_lat_15'] = test_vendor_data_merged['vendor_longitude']*np.sin(15* np.pi / 180) + test_vendor_data_merged['vendor_latitude']*np.cos(15* np.pi/180)
test_vendor_data_merged['vendor_lat_30'] = test_vendor_data_merged['vendor_longitude']*np.sin(30* np.pi / 180) + test_vendor_data_merged['vendor_latitude']*np.cos(30* np.pi/180)
test_vendor_data_merged['vendor_lat_45'] = test_vendor_data_merged['vendor_longitude']*np.sin(45* np.pi / 180) + test_vendor_data_merged['vendor_latitude']*np.cos(45* np.pi/180)
test_vendor_data_merged['vendor_lat_60'] = test_vendor_data_merged['vendor_longitude']*np.sin(60* np.pi / 180) + test_vendor_data_merged['vendor_latitude']*np.cos(60* np.pi/180)
test_vendor_data_merged['vendor_lat_75'] = test_vendor_data_merged['vendor_longitude']*np.sin(75* np.pi / 180) + test_vendor_data_merged['vendor_latitude']*np.cos(75* np.pi/180)
reduce_mem_usage(test_vendor_data_merged)
print('Rotational parameter creation ends!')
