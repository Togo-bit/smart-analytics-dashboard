from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, redirect, url_for, render_template
from config import Config
from model import db, Details
from flask_bcrypt import Bcrypt
import jwt, datetime

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

@app.route('/')
def home():
    return {"message":"App is running"}

bcrypt = Bcrypt(app)

@app.route('/register', methods = ['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = Details.query.filter_by(email=email).first()

        if existing_user:
            return {"message":"Email already exists"}

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = Details(
            username=username,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        return {'message':"User registered successfully"}

    return render_template('register.html')

@app.route('/login', methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = Details.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            token = jwt.encode({
                "user_id":user.id,
                "exp":datetime.datetime.utcnow() + datetime.timedelta(hours=2)
            }, app.config['SECRET_KEY'], algorithm='HS256')

            return {"token":token}

        return {"message":"Invalid username or password"}

    return render_template('login.html')

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
