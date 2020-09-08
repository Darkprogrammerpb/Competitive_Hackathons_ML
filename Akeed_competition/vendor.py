import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime,date,time
pd.set_option('display.max_columns', None)
import warnings
from utils import *
print('Vendor dataset handling starts !')
########### Step 1:- Loading the vendor and orders datasets ###############

order_location       = os.path.join(os.getcwd(),'raw_input_data','orders.csv')
order_df             = pd.read_csv(order_location,low_memory=False)

vendor_location      = os.path.join(os.getcwd(),'raw_input_data','vendors.csv')
vendor_df            = pd.read_csv(vendor_location,low_memory=False)


########### Step 2:- Calculating the shift timings and other necessary features for each vendor 
########### based on the type of food offered in those timings (breakfast, dinner, lunch) 

vendor_df            = vendor_df.rename(columns={'latitude':'vendor_latitude',
                                                 'longitude':'vendor_longitude',
                                                })
vendor_df['Total_hours_open_shift1']           = vendor_df['OpeningTime'].apply(total_hours_open)
vendor_df['Total_hours_open_shift2']           = vendor_df['OpeningTime2'].apply(total_hours_open)

reduce_mem_usage(vendor_df)


###### We considered only Saturdays and Sundays because only those two days were seeming to have higher feature importances and 
###### we excluded the remaining days

days = ['saturday','sunday']
for day in days:
    vendor_df['Total_hours_open_shift1_'+day]  = vendor_df[[day+'_from_time1',day+'_to_time1']].apply(lambda x:daily_hours_open(*x),axis=1)
    vendor_df['Total_hours_open_shift2_'+day]  = vendor_df[[day+'_from_time2',day+'_to_time2']].apply(lambda x:daily_hours_open(*x),axis=1)

    vendor_df['Night_service_shift1_'+day]     = vendor_df[[day+'_from_time1',day+'_to_time1']].apply(lambda x:night_service(*x),axis=1)
    vendor_df['Night_service_shift2_'+day]     = vendor_df[[day+'_from_time2',day+'_to_time2']].apply(lambda x:night_service(*x),axis=1)


    vendor_df['Breakfast_shift1_'+day]         = vendor_df[[day+'_from_time1',day+'_to_time1']].apply(lambda x:breakfast(*x),axis=1)
    vendor_df['Breakfast_shift2_'+day]         = vendor_df[[day+'_from_time2',day+'_to_time2']].apply(lambda x:breakfast(*x),axis=1)

    
    vendor_df['Lunch_snacks_shift1_'+day]      = vendor_df[[day+'_from_time1',day+'_to_time1']].apply(lambda x:lunch_snacks(*x),axis=1)
    vendor_df['Lunch_snacks_shift2_'+day]      = vendor_df[[day+'_from_time2',day+'_to_time2']].apply(lambda x:lunch_snacks(*x),axis=1)

    
    vendor_df['Dinner_shift1_'+day]            = vendor_df[[day+'_from_time1',day+'_to_time1']].apply(lambda x:dinner(*x),axis=1)
    vendor_df['Dinner_shift2_'+day]            = vendor_df[[day+'_from_time2',day+'_to_time2']].apply(lambda x:dinner(*x),axis=1)

    
    reduce_mem_usage(vendor_df)

    
vendor_df['created_at']                        = pd.to_datetime(vendor_df['created_at']).dt.date
vendor_df['updated_at']                        = pd.to_datetime(vendor_df['updated_at']).dt.date
reduce_mem_usage(vendor_df)

############### Step 3:- Creating text based features on vendor tag name to know what type of food the vendor is offering 

corpus           = vendor_df['vendor_tag_name'].fillna('no_tag').values
vectorizer       = CountVectorizer( token_pattern='(?u)[a-zA-Z][a-z ]+')
X                = vectorizer.fit_transform(corpus)
vendor_tags_data = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())


############## Step 4:- We will now drop some columns that may not make much sense in the data either because 
############## they had been used for feature creation above or are redundant.

vendor_cols_drop = [
                    ############## These are time related columns from where the features are already being extracted    
                    'sunday_from_time1', 'sunday_to_time1',
                    'sunday_from_time2', 'sunday_to_time2', 
                    'monday_from_time1','monday_to_time1',
                    'monday_from_time2', 'monday_to_time2',
                    'tuesday_from_time1', 'tuesday_to_time1', 
                    'tuesday_from_time2','tuesday_to_time2',
                    'wednesday_from_time1', 'wednesday_to_time1',
                    'wednesday_from_time2', 'wednesday_to_time2',
                    'thursday_from_time1','thursday_to_time1',
                    'thursday_from_time2', 'thursday_to_time2',
                    'friday_from_time1', 'friday_to_time1', 
                    'friday_from_time2','friday_to_time2', 
                    'saturday_from_time1', 'saturday_to_time1',
                    'saturday_from_time2', 'saturday_to_time2',
                    'OpeningTime2','OpeningTime','display_orders',
                    ######### These are redundant columns
                    'vendor_tag',  ## we already have vendor_tag_name, so it is better to drop this column
                    'country_id','city_id',  ### both the columns have just one values common accross all vendors
                    'language', ######## It's english only 
                    'authentication_id',  ### no use at all
                    'vendor_category_id', ### We are already having vencor_category_en  
                    # 'primary_tags', ## no clue what it means (will include it later)
                    'one_click_vendor','display_orders','open_close_flags','device_type','verified',
                    'is_akeed_delivering'
                   ]
            
vendor_df = vendor_df.drop(vendor_cols_drop,axis=1)

vendor_df['primary_tags']=vendor_df['primary_tags'].apply(lambda x: str(x).split(":")[-1:][0][1:2])
vendor_df = pd.concat([vendor_df, vendor_tags_data], axis=1)

####### Step 5:- Creating the same latitide and longitude based features in the data as created in
####### the customer information data 

vendor_df['vendor_manhattan_distance']   = manhattan_distance(vendor_df['vendor_latitude'].values,
                                                                                 vendor_df['vendor_longitude'].values
                                                                                )

vendor_df['vendor_Bearing_distance']     = bearing_array(vendor_df['vendor_latitude'].values,
                                                                                  vendor_df['vendor_longitude'].values
                                                                              )
vendor_df['vendor_bearing_by_manhattan'] = (vendor_df['vendor_Bearing_distance']/vendor_df['vendor_manhattan_distance'])

vendor_df=vendor_df.merge(order_df.groupby('vendor_id',as_index=False)['customer_id'].count(),how='left',left_on='id',right_on='vendor_id')
vendor_df.rename({'customer_id':'cnt_customer_id'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id',as_index=False)['promo_code'].count(),how='left',on='vendor_id')
vendor_df.rename({'promo_code':'cnt_promo_code'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id',as_index=False)['item_count'].count(),how='left',on='vendor_id')
vendor_df.rename({'item_count':'cnt_item_count'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id',as_index=False)['item_count'].mean(),how='left',on='vendor_id')
vendor_df.rename({'item_count':'mean_item_count'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id')['item_count'].std().reset_index(),how='left',on='vendor_id')
vendor_df.rename({'item_count':'std_item_count'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id',as_index=False)['grand_total'].mean(),how='left',on='vendor_id')
vendor_df.rename({'grand_total':'mean_grand_total'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id')['grand_total'].std().reset_index(),how='left',on='vendor_id')
vendor_df.rename({'grand_total':'std_grand_total'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id',as_index=False)['preparationtime'].mean(),how='left',on='vendor_id')
vendor_df.rename({'preparationtime':'mean_preparationtime'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df.groupby('vendor_id')['preparationtime'].std().reset_index(),how='left',on='vendor_id')
vendor_df.rename({'preparationtime':'std_preparationtime'},axis=1,inplace=True)


######### Step 6:- Extracting order based features from the orders data to capture more historical information 
######### about the vendor that can be used as features for each prospective customer vendor pair 

vendor_df=vendor_df.merge(order_df[order_df['is_favorite']=='Yes'].groupby('vendor_id',as_index=False)['is_favorite'].count(),how='left',on='vendor_id')
vendor_df.rename({'is_favorite':'favorite_count'},axis=1,inplace=True)

vendor_df=vendor_df.merge(order_df[order_df['is_favorite']=='No'].groupby('vendor_id',as_index=False)['is_favorite'].count(),how='left',on='vendor_id')
vendor_df.rename({'is_favorite':'not_favorite_count'},axis=1,inplace=True)
reduce_mem_usage(vendor_df)


order_vendor_merge = order_df.merge(vendor_df,how='left',left_on='vendor_id',right_on='id')
vendor_df          = vendor_df.merge(order_vendor_merge.groupby(['id']).agg({'deliverydistance':'mean'}).reset_index().rename(columns={'deliverydistance':'average_del_dist'}),
                      how='left',
                      on='id')
vendor_df          = vendor_df.merge(order_vendor_merge.groupby(['id']).agg({'deliverydistance':'std'}).reset_index().rename(columns={'deliverydistance':'std_del_dist'}),
                      how='left',
                      on='id')
      
vendor_df          = vendor_df.merge(order_vendor_merge.groupby(['id']).agg({'deliverydistance':'sum'}).reset_index().rename(columns={'deliverydistance':'Total_del_dist'}),
                      how='left',
                      on='id')

vendor_df          = vendor_df.merge(order_vendor_merge.groupby(['id']).agg({'deliverydistance':'max'}).reset_index().rename(columns={'deliverydistance':'max_del_dist'}),
                      how='left',
                      on='id')

vendor_df          = vendor_df.merge(order_vendor_merge.groupby(['id']).agg({'driver_rating':'mean'}).reset_index().rename(columns={'driver_rating':'mean_ratings_driver'}),
                      how='left',
                      on='id')

vendor_df          = vendor_df.merge(order_vendor_merge.groupby(['id']).agg({'driver_rating':'std'}).reset_index().rename(columns={'driver_rating':'std_ratings_driver'}),
                      how='left',
                      on='id')

reduce_mem_usage(vendor_df)

vendor_df['north_south_vendor']          = vendor_df['vendor_latitude'].apply(location_north_south)
vendor_df['east_west_vendor']            = vendor_df['vendor_longitude'].apply(location_east_west)
vendor_df['vendor_lat_by_long']          = vendor_df['vendor_latitude']/vendor_df['vendor_longitude']

del vendor_df['vendor_id']
reduce_mem_usage(vendor_df)
vendor_df.rename({'created_at':'vendor_created_at'},axis=1,inplace=True)
vendor_df.rename({'updated_at':'vendor_updated_at'},axis=1,inplace=True)
reduce_mem_usage(vendor_df)
vendor_df.head()


vendor_df['cv_item_count']=vendor_df['std_item_count']/vendor_df['mean_item_count']
vendor_df['cv_grand_total']=vendor_df['std_grand_total']/vendor_df['mean_grand_total']
vendor_df['cv_preparationtime']=vendor_df['std_preparationtime']/vendor_df['mean_preparationtime']
vendor_df['cv_del_dist']=vendor_df['std_del_dist']/vendor_df['average_del_dist']
vendor_df['cv_ratings_driver']=vendor_df['std_ratings_driver']/vendor_df['mean_ratings_driver']
reduce_mem_usage(vendor_df)
print('Vendor dataset handling ends !')