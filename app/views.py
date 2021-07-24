from flask import render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from app import app 
'''
app.config[" SQLALCHEMY_DATABASE_URI"] = "sqlite://users.db"
app.config["SECRET_KEY"] = "mu super"
# intitializing the database: 
db = SQLAlchemy(app)

# Creating model: 
class Users(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	message= db.Column(db.String(200),nullable=False)
	date_added = db.Column(db.DateTime,default = datetime.utcnow)

	def __repr__(self):
		return '<message %r>' %self.date_added

'''		
@app.route('/')
def home():
	return render_template('public/predicted.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

	# Training of the model....

	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

	# Features and Labels

	df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
	X = df['v2']
	y = df['v1']
	cv = CountVectorizer()
	X = cv.fit_transform(X) 
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	from sklearn.naive_bayes import MultinomialNB
    
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	print(clf.score(X_test,y_test))

	pickle.dump(X,open('vector.pkl','wb'))
	pickle.dump(clf,open('model.pkl','wb'))
	if request.method == 'POST':

		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()

		my_prediction = clf.predict(vect)

	return render_template('public/result.html',prediction = my_prediction)

@app.route("/about")
def about():
	return render_template("public/about.html")