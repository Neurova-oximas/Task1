import langclassifier
from arabictext import ArabicClassifier
from englishtext import EnglishClassifier
import nltk 
#nltk.download("punkt_tab")
#nltk.download("stopwords")
#nltk.download("wordnet")
#nltk.download("averaged_perceptron_tagger_eng")

# --- Model paths ---
ARABIC_CLASSIFIER_PATH  = r"C:\D\A-('REALLY SPECIEL')\Journeys\ML\Neurova\models\arabic_classifier_model.pkl"
ARABIC_VECTORIZER_PATH  = r"C:\D\A-('REALLY SPECIEL')\Journeys\ML\Neurova\models\arabic_vectorizer_model.pkl"
ENGLISH_CLASSIFIER_PATH = r"C:\D\A-('REALLY SPECIEL')\Journeys\ML\Neurova\models\english_classifier_model.pkl"
ENGLISH_VECTORIZER_PATH  = r"C:\D\A-('REALLY SPECIEL')\Journeys\ML\Neurova\models\english_vectorizer_model.pkl"


def main():
    text = input("Enter your text: ")

    lang = langclassifier.arabic_or_english(text)
    print(f"Language is {lang}")

    if lang == "arabic":
        classifier = ArabicClassifier(ARABIC_CLASSIFIER_PATH, ARABIC_VECTORIZER_PATH)
    else:
        classifier = EnglishClassifier(ENGLISH_CLASSIFIER_PATH,ENGLISH_VECTORIZER_PATH)
    #it could be any other language but this classifer is simple so we dont care for now
    print(f"Classifying the text...'{text}'")
    prediction = classifier.classify(text)
    print(f"The text is: {prediction}")


if __name__ == "__main__":
    while(True):
        main()
