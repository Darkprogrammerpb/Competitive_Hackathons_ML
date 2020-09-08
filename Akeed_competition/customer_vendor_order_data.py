from vendor import *
from train_customer_information import *

order_vendor_merge      = order_df.merge(vendor_df[['id','vendor_latitude','vendor_longitude']],how='left',left_on='vendor_id',right_on='id')
reduce_mem_usage(order_vendor_merge)
customer_level_data     = train_customer_information.merge(order_vendor_merge,how='left',on='customer_id')

customer_level_data['haversine_distance_covered'] = haversine_distance(customer_level_data,
                                                                       'vendor_latitude', 'vendor_longitude',
                                                                       'cust_latitude', 'cust_longitude')
reduce_mem_usage(customer_level_data)
customer_level_data['cust_lat_by_vendor_lat']   = customer_level_data['cust_latitude']/customer_level_data['vendor_latitude']
customer_level_data['cust_lat_by_vendor_long']  = customer_level_data['cust_latitude']/customer_level_data['vendor_longitude']
customer_level_data['cust_long_by_vendor_lat']  = customer_level_data['cust_longitude']/customer_level_data['vendor_latitude']
customer_level_data['cust_long_by_vendor_long'] = customer_level_data['cust_longitude']/customer_level_data['vendor_longitude']

reduce_mem_usage(customer_level_data)
total_haversine_covered = customer_level_data.groupby(['id']).agg({'haversine_distance_covered':'sum'}).reset_index().rename(columns={'haversine_distance_covered':'total_haversine_covered_vendor'})
mean_haversine_covered  = customer_level_data.groupby(['id']).agg({'haversine_distance_covered':'mean'}).reset_index().rename(columns={'haversine_distance_covered':'mean_haversine_covered_vendor'})
std_haversine_covered   = customer_level_data.groupby(['id']).agg({'haversine_distance_covered':'std'}).reset_index().rename(columns={'haversine_distance_covered':'std_haversine_covered_vendor'})
reduce_mem_usage(total_haversine_covered)
reduce_mem_usage(mean_haversine_covered)
reduce_mem_usage(std_haversine_covered)


vendor_df               = data_left_join(vendor_df,total_haversine_covered,['id'])
reduce_mem_usage(vendor_df)
vendor_df               = data_left_join(vendor_df,mean_haversine_covered,['id'])
reduce_mem_usage(vendor_df)
vendor_df               = data_left_join(vendor_df,std_haversine_covered,['id'])
reduce_mem_usage(vendor_df)

# vendor_df               = data_left_join(vendor_df,customer_level_data[['id','cust_lat_by_vendor_lat']],['id'])
# vendor_df               = data_left_join(vendor_df,customer_level_data[['id','cust_lat_by_vendor_long']],['id'])
# reduce_mem_usage(vendor_df)
# vendor_df               = data_left_join(vendor_df,customer_level_data[['id','cust_long_by_vendor_lat']],['id'])
# vendor_df               = data_left_join(vendor_df,customer_level_data[['id','cust_long_by_vendor_long']],['id'])
#reduce_mem_usage(vendor_df)
