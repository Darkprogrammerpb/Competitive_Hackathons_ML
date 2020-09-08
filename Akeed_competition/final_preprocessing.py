from create_test_train_data import *
from sklearn.cluster import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import MiniBatchDictionaryLearning


train_customer_information = reduce_mem_usage(train_customer_information)
test_customer_information  = reduce_mem_usage(test_customer_information)
print('Clustering for customer-vendor combination started!')
################## Clustering based on concatenated dataset(customer-vendor combinations) #####################################
num_clusters                        = 50
clus                                = KMeans(n_clusters=num_clusters,random_state=202020)
concat_df                           = pd.concat((train_vendor_data_merged[['cust_latitude','cust_longitude','vendor_latitude','vendor_longitude']].fillna(999),test_vendor_data_merged[['cust_latitude','cust_longitude','vendor_latitude','vendor_longitude']].fillna(999)),axis=0)

clus.fit(concat_df.values)

labels                              = clus.labels_
train_labels                        = labels[:train_vendor_data_merged.shape[0]]
test_labels                         = labels[train_vendor_data_merged.shape[0]:]
train_vendor_data_merged['cluster_label_overall'] = train_labels
test_vendor_data_merged['cluster_label_overall']  = test_labels

reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)

print('Clustering for customer-vendor combination ends!')

print('Clustering for customer vendor considered independent started!')
##################### Clustering performed when vendors and customers are considered independent ##############################
coords = np.vstack((train_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999).values,
                    train_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999).values,
                    test_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999).values,
                    test_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999).values))

np.random.seed(202020)
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000,random_state=202020).fit(coords[sample_ind])

train_vendor_data_merged.loc[:, 'cutomer_cluster'] = kmeans.predict(train_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999))
train_vendor_data_merged.loc[:, 'vendor_cluster'] = kmeans.predict(train_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999))

test_vendor_data_merged.loc[:, 'cutomer_cluster'] = kmeans.predict(test_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999))
test_vendor_data_merged.loc[:, 'vendor_cluster'] = kmeans.predict(test_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999))

reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)
print('Clustering for customer vendor considered independent ended!')


########################## PCA to capture the variability component for customer and vendor lcoations #########################
coords = np.vstack((train_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999).values,
                    train_vendor_data_merged[['vendor_latitude', 'vendor_latitude']].fillna(999).values,
                    test_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999).values,
                    test_vendor_data_merged[['vendor_latitude', 'vendor_latitude']].fillna(999).values))
pca = PCA(random_state=202020).fit(coords)
train_vendor_data_merged['cust_pca1'] = pca.transform(train_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999))[:, 0]
train_vendor_data_merged['cust_pca2'] = pca.transform(train_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999))[:, 1]
train_vendor_data_merged['vendor_pca1'] = pca.transform(train_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999))[:, 0]
train_vendor_data_merged['vendor_pca2'] = pca.transform(train_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999))[:, 1]


test_vendor_data_merged['cust_pca1'] = pca.transform(test_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999))[:, 0]
test_vendor_data_merged['cust_pca2'] = pca.transform(test_vendor_data_merged[['cust_latitude', 'cust_longitude']].fillna(999))[:, 1]
test_vendor_data_merged['vendor_pca1'] = pca.transform(test_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999))[:, 0]
test_vendor_data_merged['vendor_pca2'] = pca.transform(test_vendor_data_merged[['vendor_latitude', 'vendor_longitude']].fillna(999))[:, 1]
 
reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)
print('PCA done for customers and vendor locations')
############################################# PCA ENDS ##############################################

################################# SOme other ratio transforms #############################################
train_vendor_data_merged['ratio_latitude_longitude']=train_vendor_data_merged['cust_latitude']/train_vendor_data_merged['cust_longitude']

test_vendor_data_merged['ratio_latitude_longitude']=test_vendor_data_merged['cust_latitude']/test_vendor_data_merged['cust_longitude']

train_vendor_data_merged['ratio_vendor_latitude_vendor_longitude']=train_vendor_data_merged['vendor_latitude']/train_vendor_data_merged['vendor_longitude']

test_vendor_data_merged['ratio_vendor_latitude_vendor_longitude']=test_vendor_data_merged['vendor_latitude']/test_vendor_data_merged['vendor_longitude']

reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)
##############################################################################################################


############################################# Online dict learning for food recipies #################################
print('dictioary learning for food items started !')
cols=['american',
 'arabic',
 'asian',
 'bagels',
 'biryani',
 'breakfast',
 'burgers',
 'cafe',
 'cakes',
 'chinese',
 'churros',
 'coffee',
 'combos',
 'crepes',
 'desserts',
 'dimsum',
 'donuts',
 'family meal',
 'fatayers',
 #'free delivery',
 'fresh juices',
 'fries',
 'frozen yoghurt',
 'grills',
 'healthy food',
 'hot chocolate',
 'hot dogs',
 'ice creams',
 'indian',
 'italian',
 'japanese',
 'karak',
 'kebabs',
 'kids meal',
 'kushari',
 'lebanese',
 'manakeesh',
 'mandazi',
 'mexican',
 'milkshakes',
 'mishkak',
 'mojitos',
 'mojitos ',
 'no',
 'omani',
 'organic',
 'pancakes',
 'pasta',
 'pastas',
 'pastry',
 'pizza',
 'pizzas',
 'rice',
 'rolls',
 'salads',
 'sandwiches',
 'seafood',
 'shawarma',
 'shuwa',
 'smoothies',
 'soups',
 'spanish latte',
 'steaks',
 'sushi',
 'sweets',
 'tag',
 'thai',
 'thali',
 'vegetarian',
 'waffles',
]

recipies_train = train_vendor_data_merged[cols]
recipies_test  = test_vendor_data_merged[cols]
concat_df      = pd.concat([recipies_train,recipies_test],axis=0)
coords         = concat_df.values

mbd = MiniBatchDictionaryLearning(n_components=1,transform_algorithm='omp',random_state=202020).fit(coords)
train_vendor_data_merged['od1'] = mbd.transform(train_vendor_data_merged[cols])[:, 0]
test_vendor_data_merged['od1']  = mbd.transform(test_vendor_data_merged[cols])[:, 0]

reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)
print('dictioary learning for food items ends !')
#############################################################################################################

################### Dropping redundant columns ##############
cols_to_drop = cols+['customer_id','dob']+['std_haversine_covered_vendor','total_haversine_covered_vendor']


train_vendor_data_merged = train_vendor_data_merged.drop(cols_to_drop,axis=1)
test_vendor_data_merged  = test_vendor_data_merged.drop(cols_to_drop,axis=1)
reduce_mem_usage(train_vendor_data_merged)
reduce_mem_usage(test_vendor_data_merged)


print('Train and test datasets created !!! ')



