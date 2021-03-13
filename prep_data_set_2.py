import pandas as pd
import boto3
from sklearn.model_selection import train_test_split

class prep_data:
    def __init__(self,dataset,bucketname,modelname):
        self.dataset = dataset
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.bucketname = bucketname
        self.modelname = modelname
        s3 = boto3.client('s3')
        s3.put_object(Bucket=self.bucketname, Key=('models/'+str(self.modelname)+'/'))
        #self.directory = 's3://{}/models/{}'.format(bucketname,modelname)
    def split(self,splitRatio):
        dict = {"fly_ash":2,"age":7,"coarse_aggregate":5,"cement":0,"water":3,"fine_aggregate":6,"blast_furnace_slag":1,"compressive_strength":8,"superplasticizer":4}
        #Name of the s3 bucket which contains the dataset
        #Path to the s3 bucket location which contains the dataset
        data_location = 's3://{}/cement_dataset/Version/{}'.format(self.bucketname, self.dataset) 
        df = pd.read_csv(data_location)
        #print(df)
        concrete_data = df.copy()
        for i in dict:
            concrete_data.iloc[:,dict[i]]=df[i]
            concrete_data
        concrete_data.drop(concrete_data.columns[9], axis=1, inplace=True)
        col_name = ["cement","blast_furnace_slag","fly_ash","water","superplasticizer","coarse_aggregate","fine_aggregate","age","compressive_strength"]
        concrete_data.columns = col_name
        #print(concrete_data)
        #concrete_data = pd.read_csv('cement.csv')
        min_d = concrete_data.min()
        max_d = concrete_data.max()
        a = []
        for i in range(0,min_d.shape[0]):
            a.append([min_d[i],max_d[i]])    
        normalized_df=(concrete_data - min_d)/(max_d - min_d)
        limit = str(a)
        text_file = open("limits.txt", "wt")
        n = text_file.write(limit)
        text_file.close()
        normal_train = normalized_df.iloc[:,:8]
        normal_label = normalized_df.iloc[:,-1:]
        print(normal_train)
        print(normal_label)
        #Spliting the dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(normal_train, normal_label, test_size = splitRatio, random_state = 1)
        self.x_train.to_csv("x_train.csv")
        self.y_train.to_csv("y_train.csv")
        self.x_test.to_csv("x_test.csv")
        self.y_test.to_csv("y_test.csv")
        s3 = boto3.resource('s3')
        print("starting upload")
        s3.meta.client.upload_file("x_train.csv", self.bucketname,'models/{}/{}'.format(self.modelname,'x_train.csv'))
        s3.meta.client.upload_file("y_train.csv", self.bucketname,'models/{}/{}'.format(self.modelname,'y_train.csv'))
        s3.meta.client.upload_file("x_test.csv", self.bucketname,'models/{}/{}'.format(self.modelname,'x_test.csv'))
        s3.meta.client.upload_file("y_test.csv", self.bucketname,'models/{}/{}'.format(self.modelname,'y_test.csv'))
        s3.meta.client.upload_file("limits.txt", self.bucketname,'models/{}/{}'.format(self.modelname,'limits.txt'))
        text_file = open("model_info.txt", "wt")
        n = text_file.write(self.modelname)
        text_file.close()
        s3.meta.client.upload_file("model_info.txt", self.bucketname,'models/{}/{}'.format(self.modelname,'model_info.txt'))
        print("uploaded")
