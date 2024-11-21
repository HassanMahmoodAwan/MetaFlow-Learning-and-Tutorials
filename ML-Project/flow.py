from metaflow import FlowSpec, step, card, pypi_base, Parameter, retry, catch, current
import os
from metaflow.cards import Image


@pypi_base(
    packages={
        "scikit-learn": "1.5.2",
        "pandas": "2.2.2",
        "numpy": "2.1.3",
        "scipy": "1.14.1",
        "matplotlib": "3.9.2",
        "seaborn": "0.13.2",
    }
)
class MachineLearningProject(FlowSpec):
    
    # *********** PARAMETERS *************
    dataset_path = Parameter("dataset_path", help="Dataset Path CSV FILE", default=os.path.abspath("ML-Project/Dataset/heart_disease_uci.csv"))
    
    
    
    @card
    @step
    def start(self):
        # pylint: disable=import-error,no-member
        import pandas as pd
        
        self.error = None
        
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset_shape = self.dataset.shape
        self.duplicate_rows = self.dataset.duplicated().sum()
        self.num_null_columns = self.dataset.isnull().sum()
        
        self.next(self.dataset_processing)
    
    
    
    @card
    @retry
    @catch
    @step
    def dataset_processing(self):
        
        # Removing Columns with larger Null Values
        self.dataset.drop(["ca", "thal"], axis=1, inplace=True)
        
        # Filling Null Values with mean value.
        self.dataset["oldpeak"].fillna(self.dataset["oldpeak"].mean(),  inplace = True)
        self.dataset["thalch"].fillna(self.dataset["thalch"].mean(),  inplace = True)
        self.dataset["trestbps"].fillna(self.dataset["trestbps"].mean(),  inplace = True)
        self.dataset["chol"].fillna(self.dataset["chol"].mean(),  inplace = True)
        self.dataset["fbs"].fillna(False,  inplace = True)
        self.dataset["restecg"].fillna("normal",  inplace = True)
        self.dataset["exang"].fillna(False,  inplace = True)
        self.dataset["slope"].fillna("flat",  inplace = True)
        
        self.dataset = self.dataset
        self.dataset_shape = self.dataset.shape
        self.duplicate_rows = self.dataset.duplicated().sum()
        self.num_null_columns = self.dataset.isnull().sum()
        
        self.next(self.feature_engineering)
        
    
    
    @card
    @retry
    @catch
    @step
    def feature_engineering(self):
        import pandas as pd
        
        # Boolean into Int
        self.dataset["fbs"] = self.dataset["fbs"].astype("int")
        self.dataset["exang"] = self.dataset["exang"].astype("int")

    
        
        # Object features into Numerical Encoded features
        temp_df = self.dataset
        
        labels=['asymptomatic', 'non-anginal', 'atypical angina', 'typical angina']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["cp"] = self.dataset["cp"].map(mapping)
        
        labels=['Male', 'Female']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["sex"] = self.dataset["sex"].map(mapping)
        
        labels=['downsloping', 'flat', 'upsloping']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["slope"] = self.dataset["slope"].map(mapping)
        
        labels = ['lv hypertrophy', 'normal', 'st-t abnormality']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["restecg Encoded"] = self.dataset["restecg"].map(mapping)

        
        # One-hot Encoding on Categorical Data
        temp_df = pd.get_dummies(temp_df, columns=["sex", "exang", 'fbs', 'cp', 'slope'])
        
        print(temp_df.head(5))
        self.preparedData = temp_df

        
        self.next(self.model_training_preprocess)
        
    
    @card
    @step
    def model_training_preprocess(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        
        
        # current.card.append(Image.from_matplotlib(
        #     self.heatmap
        # ))
        
        matrix = show_matrix(self.preparedData)
        current.card.append(Image.from_matplotlib(matrix))
        plt.close(matrix)
        
        # Training and Testing Data
        self.X = self.preparedData.drop(['id', 'dataset', 'restecg' ,'restecg Encoded'], axis = 1)
        self.Y = self.preparedData['restecg']
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=10)
        
        self.models = ["logisticRegression", "decisionTree", "RandomForest"]
        
        self.next(self.model_training, foreach = "models")
        
    
    @card
    # @kubernetes                  :: Configure Cloud First to Run.
    @retry
    @catch
    @step
    def model_training(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix

        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        
        
        if self.input == "logisticRegression":
            self.model = LogisticRegression()
            
            self.target =  (self.preparedData["restecg"] == 'normal').astype(int)
            x_train, x_test, target_train, target_test = train_test_split(self.X, self.target, test_size=0.3, random_state=10)
            
            self.model.fit(x_train, target_train)
            target_pred = self.model.predict(x_test)
            
            self.Accuracy = accuracy_score(target_pred, target_test)
            print(self.Accuracy)

        elif self.input == "decisionTree":
            self.model = DecisionTreeClassifier()

            self.model.fit(self.X_train, self.Y_train)
            Y_pred = self.model.predict(self.X_test)
            self.Accuracy = accuracy_score(Y_pred, self.Y_test)
            print(self.Accuracy)
            
            
        elif self.input == "RandomForest":
            self.model = RandomForestClassifier(n_estimators = 100)
            self.model.fit(self.X_train, self.Y_train)
            
            Y_pred = self.model.predict(self.X_test)
            self.Accuracy = accuracy_score(self.Y_test, Y_pred)
            
            print(self.Accuracy)
            
        self.next(self.join)
        
        
    
    @card
    @step
    def join(self, inputs):
        import pandas as pd
        
        models = ["Logistic Regression", "Decision Tree", "Random Forest"]
        accuracy = [i.Accuracy for i in inputs]
        
        temp = {"Models": models, "Accuracy": accuracy} 
        self.Accuracy = pd.DataFrame(temp)
        self.next(self.end)

        
    @card
    @step
    def end(self):
        print(self.Accuracy)




def show_matrix(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import copy
    
    dataframe = copy.deepcopy(df)
    
    dataframe.drop("dataset", axis=1, inplace=True)
    dataframe.drop("restecg", axis=1, inplace=True)
    plt.figure(figsize=(10, 8))
    matrix = dataframe.corr()
    sns.heatmap(matrix, annot= True, cmap="coolwarm", fmt=".2f")
    return plt.gcf()
    
    
    
    
    



if __name__ == "__main__":
    MachineLearningProject()