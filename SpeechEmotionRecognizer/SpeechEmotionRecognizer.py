from abc import abstractmethod
import librosa

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
    def test(self):
        pass

    @abstractmethod
    def predict(self):
        pass

