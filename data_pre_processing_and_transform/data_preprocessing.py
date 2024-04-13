import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessing():
    def __init__(self, directory, independent_signals, dependent_signals, sequence_length, train_size=.9, do_train_test_split = True):
        
      self.directory = directory
      self.do_train_test_split = do_train_test_split
      self.list_of_path = DataPreprocessing.list_csv_files(directory)
      self.train_size = train_size
      self.independent_signals = independent_signals
      self.dependent_signals = dependent_signals
      self.sequence_length = sequence_length

      self.train_df_batch = []
      self.test_df_batch = []
      self.dataset_batch = []
      self.train_sequences = []
      self.test_sequences = []

      self.scaler = MinMaxScaler(feature_range=(-1,1))
    
    @staticmethod
    def split_dataset(data, train_size):
      train_size = int(len(data)* train_size)
      train_df, test_df = data[:train_size], data[train_size + 1:]
      return train_df, test_df  

    @staticmethod
    def list_csv_files(directory):
      csv_files = []
      for filename in os.listdir(directory):
        if filename.endswith(".csv"):
          csv_files.append(filename)
      return csv_files    
    
    @staticmethod
    def load_data_from_a_specific_path(file_path):
      dataset = pd.read_csv(file_path)
      dataset.timestamp = pd.to_datetime(dataset.timestamp)
      dataset = dataset.set_index('timestamp')
      return dataset
    
    
    def scaling_fit(self):
      self.scaler.fit(pd.concat(self.train_df_batch))

    def scaling_transform(self, dataset):
      return pd.DataFrame(self.scaler.transform(dataset),
                          index=dataset.index,
                          columns=dataset.columns)
    
    def data_loader(self, file_path):

      path = '{}{}'.format(self.directory, file_path) 
      print(path)
      dataset = pd.read_csv(path)
      dataset.timestamp = pd.to_datetime(dataset.timestamp)
      dataset = dataset.set_index('timestamp')
      return dataset
    
    def fit_transform(self):  
      for each_file_path in self.list_of_path:
        # Loading data from each file
        dataset = self.data_loader(each_file_path)

        # Calculate min and max value of all features this will be used for scaling
        self.dataset_batch.append(dataset)
        
        train_df, test_df = DataPreprocessing.split_dataset(dataset, self.train_size)
        self.train_df_batch.append(train_df)
        self.test_df_batch.append(test_df)

      # Scaling
      self.scaling_fit()
      
      # Scaling transform
      for batch_idx in range(len(self.train_df_batch)):
        self.train_df_batch[batch_idx] = (self.scaling_transform(self.train_df_batch[batch_idx]))
        self.test_df_batch[batch_idx] = self.scaling_transform(self.test_df_batch[batch_idx])


      # For train data
      self.create_sequences(self.sequence_length)
      # For test data
      self.create_sequences(self.sequence_length, get_sequences_for_train = False)


    def transform(self, path):
        dataset = DataPreprocessing.load_data_from_a_specific_path(path)
        self.datset_sequences = []
        scaled_dataset = (self.scaling_transform(dataset))
        self.create_sequences_for_a_batch(scaled_dataset, self.sequence_length, None)
        return self.datset_sequences


    def create_sequences_for_a_batch(self, input_data, sequence_length, get_sequences_for_train):

      data_size = len(input_data)

      for i in range(data_size - sequence_length):

        sequence = input_data[i: i+sequence_length][self.independent_signals]

        if get_sequences_for_train: 
          label = input_data[i: i + sequence_length][self.dependent_signals]
          self.train_sequences.append((sequence, label))

        elif get_sequences_for_train ==None:
          self.datset_sequences.append((sequence, None)) 

        else:
          label = input_data[i: i + sequence_length][self.dependent_signals]
          self.test_sequences.append((sequence,label))  


    def create_sequences(self, sequence_length, get_sequences_for_train=True):

      if get_sequences_for_train:

        for each_batch in self.train_df_batch:
          self.create_sequences_for_a_batch(each_batch, sequence_length, get_sequences_for_train)

      else:

        for each_batch in self.test_df_batch:
          self.create_sequences_for_a_batch(each_batch, sequence_length, get_sequences_for_train)  
        

    # Getter method
    @property
    def get_train_df(self):
      return self.train_df_batch

    @property
    def get_test_df(self):
      return self.test_df_batch 
    
    @property
    def get_sequences_train(self):
      return self.train_sequences
    
    @property
    def get_sequences_test(self):
      return self.test_sequences  
    
