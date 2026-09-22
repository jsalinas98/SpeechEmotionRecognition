from abc import abstractmethod
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SpeechEmotionRecognizer():

    def __init__(self):
        self.audios = []
        self.labels = []
        self.X = []
        self.Y = []
        self.TrainValidationSplit = 0.2
        self.sampleRate = 0
        
    def loadData(self, audioPaths, labels):
        assert len(audioPaths) == len(labels), "Data length is inconsistent"
        i=0
        for index1, path in enumerate(audioPaths):
            print('loading data: ', i,'/',len(audioPaths), end="\r")
            X, self.sampleRate = librosa.load(path, duration=3, offset=0)
            self.audios.append(X)
            i+=1 

        for index2, label in enumerate(labels):
            self.labels.append(label)

        print('\n')
        print('data loaded correctly!')

    @abstractmethod
    def dataProcess(self):
        pass

    @abstractmethod
    def createModel(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def test(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        return accuracy_score(y_true=y_test, y_pred=y_pred)
    
    def printConfusionMatrix(self, y_test, y_pred, categories):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize = (12, 10))     
        cm = pd.DataFrame(cm , index = [i for i in categories] , columns = [i for i in categories])
        sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.show()
