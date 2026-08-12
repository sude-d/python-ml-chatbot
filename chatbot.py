from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
mesajlar = [

    # ==================== GREETING ====================
    "Hello",
    "Hi",
    "Hey",
    "Hello there",
    "Hi there",
    "Hey there",
    "Good morning",
    "Good afternoon",
    "Good evening",
    "Nice to meet you",

    # ==================== HOW ARE YOU ====================
    "How are you?",
    "How are you doing?",
    "How have you been?",
    "Are you doing well?",
    "How is everything?",
    "How are things going?",
    "Are you okay?",
    "How do you feel?",
    "Is everything going well?",
    "How has your day been?",

    # ==================== ABOUT BOT ====================
    "Who are you?",
    "What are you?",
    "Can you tell me about yourself?",
    "What should I call you?",
    "Who am I talking to?",
    "Tell me who you are",
    "Can you introduce yourself?",
    "What is your name?",
    "I want to know who you are",
    "Can you tell me your name?",

    # ==================== CAPABILITIES ====================
    "What can you do?",
    "What are you capable of?",
    "How can you help me?",
    "What can you help me with?",
    "What kind of questions can I ask?",
    "What do you know about?",
    "What can I ask you?",
    "What are your abilities?",
    "How can I use you?",
    "What can you help me learn?",

    # ==================== THANKS ====================
    "Thank you",
    "Thanks",
    "Thank you very much",
    "Thanks for your help",
    "I appreciate your help",
    "That was helpful",
    "You helped me a lot",
    "Thanks for explaining",
    "I appreciate it",
    "Thank you for the answer",

    # ==================== GOODBYE ====================
    "Goodbye",
    "Bye",
    "See you later",
    "See you soon",
    "Talk to you later",
    "I have to go",
    "I need to leave",
    "Bye for now",
    "See you next time",
    "Have a nice day",

    # ==================== NLP ====================
    "What is NLP?",
    "What does NLP mean?",
    "Can you explain NLP?",
    "What is natural language processing?",
    "How does NLP work?",
    "What is natural language processing used for?",
    "Why is NLP important?",
    "Where is NLP used?",
    "What can NLP do?",
    "What are NLP applications?",

    # ==================== PYTHON ====================
    "What is Python?",
    "What is Python used for?",
    "Why is Python popular?",
    "Why should I learn Python?",
    "Can you explain Python?",
    "Is Python good for beginners?",
    "What can I build with Python?",
    "Is Python useful for AI?",
    "Can Python be used for machine learning?",
    "What are the advantages of Python?",

    # ==================== NUMPY ====================
    "What is NumPy?",
    "What does NumPy do?",
    "What is NumPy used for?",
    "Can you explain NumPy?",
    "Why is NumPy useful?",
    "How do I use NumPy?",
    "What is a NumPy array?",
    "How do I create a NumPy array?",
    "Can NumPy work with matrices?",
    "Why do people use NumPy?",

    # ==================== PANDAS ====================
    "What is Pandas?",
    "What is pandas used for?",
    "Can you explain pandas?",
    "Why is pandas useful?",
    "How do I use pandas?",
    "What is a DataFrame?",
    "What is a pandas DataFrame?",
    "How do I create a DataFrame?",
    "Can pandas analyze data?",
    "Why do data scientists use pandas?",

    # ==================== SKLEARN ====================
    "What is scikit-learn?",
    "What is sklearn?",
    "What is scikit-learn used for?",
    "Can you explain sklearn?",
    "Why is scikit-learn useful?",
    "How do I use sklearn?",
    "Is sklearn a machine learning library?",
    "Can sklearn build classifiers?",
    "Can sklearn be used for NLP?",
    "What algorithms does sklearn provide?",

    # ==================== MACHINE LEARNING ====================
    "What is machine learning?",
    "Can you explain machine learning?",
    "How does machine learning work?",
    "What is machine learning used for?",
    "Why is machine learning important?",
    "What are the types of machine learning?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is a machine learning model?",
    "How does a machine learning model learn?",

    # ==================== AI ====================
    "What is artificial intelligence?",
    "What does AI mean?",
    "Can you explain artificial intelligence?",
    "How does AI work?",
    "What is AI used for?",
    "Why is artificial intelligence important?",
    "Where is AI used?",
    "What are examples of AI?",
    "Can AI learn from data?",
    "What are the main areas of AI?",

    # ==================== DEEP LEARNING ====================
    "What is deep learning?",
    "Can you explain deep learning?",
    "How does deep learning work?",
    "What is a neural network?",
    "What are neural networks used for?",
    "What is an artificial neural network?",
    "Why is deep learning important?",
    "Where is deep learning used?",
    "What is a deep learning model?",
    "Is deep learning part of machine learning?",

    # ==================== CLASSIFICATION ====================
    "What is classification?",
    "What is classification in machine learning?",
    "How does classification work?",
    "What is a classification model?",
    "What is a classifier?",
    "How can I classify data?",
    "What are classification algorithms?",
    "Can machine learning classify text?",
    "What is binary classification?",
    "What is multiclass classification?",

    # ==================== REGRESSION ====================
    "What is regression?",
    "What is regression in machine learning?",
    "How does regression work?",
    "What is a regression model?",
    "When is regression used?",
    "Can regression predict numbers?",
    "What are regression algorithms?",
    "What is linear regression?",
    "How does linear regression work?",
    "What is the difference between classification and regression?",

    # ==================== TRAINING ====================
    "What is training data?",
    "What is a training dataset?",
    "How do I train a model?",
    "What does model training mean?",
    "Why do models need training data?",
    "How does training work in machine learning?",
    "What happens during model training?",
    "How long does model training take?",
    "What is a training set?",
    "Why is training important?",

    # ==================== PREDICTION ====================
    "What is a prediction?",
    "How does a model make predictions?",
    "Can machine learning make predictions?",
    "How do I make a prediction with a model?",
    "What does model prediction mean?",
    "Why do models make predictions?",
    "How accurate can predictions be?",
    "Can AI predict future values?",
    "What is predictive modeling?",
    "How does prediction work in machine learning?",

    # ==================== PYTHON HELP ====================
    "How do I create a Python function?",
    "How do Python loops work?",
    "How do I create a list in Python?",
    "How do I create a dictionary in Python?",
    "How do I use an if statement in Python?",
    "How do I install a Python package?",
    "How do I import a library in Python?",
    "How do I read a file in Python?",
    "How do I handle errors in Python?",
    "How do I run a Python program?"
]


etiketler = [

    # greeting - 10
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",

    # how_are_you - 10
    "how_are_you",
    "how_are_you",
    "how_are_you",
    "how_are_you",
    "how_are_you",
    "how_are_you",
    "how_are_you",
    "how_are_you",
    "how_are_you",
    "how_are_you",

    # about_bot - 10
    "about_bot",
    "about_bot",
    "about_bot",
    "about_bot",
    "about_bot",
    "about_bot",
    "about_bot",
    "about_bot",
    "about_bot",
    "about_bot",

    # capabilities - 10
    "capabilities",
    "capabilities",
    "capabilities",
    "capabilities",
    "capabilities",
    "capabilities",
    "capabilities",
    "capabilities",
    "capabilities",
    "capabilities",

    # thanks - 10
    "thanks",
    "thanks",
    "thanks",
    "thanks",
    "thanks",
    "thanks",
    "thanks",
    "thanks",
    "thanks",
    "thanks",

    # goodbye - 10
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",
    "goodbye",

    # nlp - 10
    "nlp",
    "nlp",
    "nlp",
    "nlp",
    "nlp",
    "nlp",
    "nlp",
    "nlp",
    "nlp",
    "nlp",

    # python - 10
    "python",
    "python",
    "python",
    "python",
    "python",
    "python",
    "python",
    "python",
    "python",
    "python",

    # numpy - 10
    "numpy",
    "numpy",
    "numpy",
    "numpy",
    "numpy",
    "numpy",
    "numpy",
    "numpy",
    "numpy",
    "numpy",

    # pandas - 10
    "pandas",
    "pandas",
    "pandas",
    "pandas",
    "pandas",
    "pandas",
    "pandas",
    "pandas",
    "pandas",
    "pandas",

    # sklearn - 10
    "sklearn",
    "sklearn",
    "sklearn",
    "sklearn",
    "sklearn",
    "sklearn",
    "sklearn",
    "sklearn",
    "sklearn",
    "sklearn",

    # machine_learning - 10
    "machine_learning",
    "machine_learning",
    "machine_learning",
    "machine_learning",
    "machine_learning",
    "machine_learning",
    "machine_learning",
    "machine_learning",
    "machine_learning",
    "machine_learning",

    # ai - 10
    "ai",
    "ai",
    "ai",
    "ai",
    "ai",
    "ai",
    "ai",
    "ai",
    "ai",
    "ai",

    # deep_learning - 10
    "deep_learning",
    "deep_learning",
    "deep_learning",
    "deep_learning",
    "deep_learning",
    "deep_learning",
    "deep_learning",
    "deep_learning",
    "deep_learning",
    "deep_learning",

    # classification - 10
    "classification",
    "classification",
    "classification",
    "classification",
    "classification",
    "classification",
    "classification",
    "classification",
    "classification",
    "classification",

    # regression - 10
    "regression",
    "regression",
    "regression",
    "regression",
    "regression",
    "regression",
    "regression",
    "regression",
    "regression",
    "regression",

    # training - 10
    "training",
    "training",
    "training",
    "training",
    "training",
    "training",
    "training",
    "training",
    "training",
    "training",

    # prediction - 10
    "prediction",
    "prediction",
    "prediction",
    "prediction",
    "prediction",
    "prediction",
    "prediction",
    "prediction",
    "prediction",
    "prediction",

    # python_help - 10
    "python_help",
    "python_help",
    "python_help",
    "python_help",
    "python_help",
    "python_help",
    "python_help",
    "python_help",
    "python_help",
    "python_help"
]
cevaplar = {

    "greeting":
        "Hello! Nice to meet you.",

    "how_are_you":
        "I'm doing great, thank you for asking! How are you?",

    "about_bot":
        "I'm a chatbot built with Python and machine learning.",

    "capabilities":
        "I can answer questions about Python, AI, NLP and machine learning.",

    "thanks":
        "You're welcome! I'm happy to help.",

    "goodbye":
        "Goodbye! Have a great day!",

    "nlp":
        "NLP stands for Natural Language Processing. It helps computers work with human language.",

    "python":
        "Python is a popular programming language used in areas such as AI, data science and web development.",

    "numpy":
        "NumPy is a Python library mainly used for numerical computing and working with arrays.",

    "pandas":
        "Pandas is a Python library used for data analysis and working with structured data.",

    "sklearn":
        "Scikit-learn is a Python machine learning library that provides many algorithms and tools.",

    "machine_learning":
        "Machine learning allows computers to learn patterns from data and make predictions.",

    "ai":
        "Artificial intelligence is the field of creating systems that can perform tasks requiring human-like intelligence.",

    "deep_learning":
        "Deep learning is a branch of machine learning that uses neural networks with multiple layers.",

    "classification":
        "Classification is a machine learning task where data is assigned to predefined categories.",

    "regression":
        "Regression is used to predict continuous numerical values.",

    "training":
        "Training is the process of teaching a machine learning model using data.",

    "prediction":
        "A prediction is the output produced by a trained machine learning model for new data.",

    "python_help":
        "I can help you with Python basics, functions, loops, lists, dictionaries and libraries."
}

model = Pipeline([
    ("metnisayiyadonustur",TfidfVectorizer()),
    ("lojistikregresyon",LogisticRegression(random_state=37,))
])
#max_iter=1000 kaç tekrar edileceğini belirler. Eğer modelinizin eğitimi sırasında "ConvergenceWarning" hatası alıyorsak, bu değeri artıracağız.
#random_state=37, modelin her çalıştırıldığında aynı sonuçları üretmesini sağlar. Bu, modelin tekrarlanabilirliğini artırır ve sonuçların tutarlılığını sağlar.
print("How can I help you?( Type 'exit' to quit)")
model.fit(mesajlar, etiketler)
while True:
    kullaniciMesaji = input("You: ")
    if kullaniciMesaji.lower() == "exit":
        print("Remember to take care of yourself, you are unique!")
        break
    tahmin = model.predict([kullaniciMesaji])
    response = cevaplar.get(tahmin[0], "I'm sorry, I don't understand your question.")
    print("ChaddBot:", response)


