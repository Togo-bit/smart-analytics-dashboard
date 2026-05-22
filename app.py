from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, redirect, url_for, render_template
from config import Config
from model import db, Details
from flask_bcrypt import Bcrypt
import jwt, datetime
import os
from authlib.integrations.flask_client import OAuth
from flask import session, jsonify

app = Flask(__name__)
app.config.from_object(Config)

app.secret_key = app.config['SECRET_KEY']

oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url=
    'https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

db.init_app(app)

@app.route('/')
def home():
    return {"message":"App is running"}

@app.route('/google/login')
def google_login():

    redirect_uri = url_for(
        'google_callback',
        _external=True
    )

    return google.authorize_redirect(redirect_uri)

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

@app.route('/google/callback')
def google_callback():

    token = google.authorize_access_token()

    user_info = token.get('userinfo')

    email = user_info['email']
    username = user_info['name']

    # CHECK IF USER EXISTS
    user = Details.query.filter_by(email=email).first()

    # CREATE NEW USER IF NOT EXISTS
    if not user:

        user = Details(
            username=username,
            email=email,
            password="google_oauth"
        )

        db.session.add(user)
        db.session.commit()

    # CREATE JWT TOKEN
    jwt_token = jwt.encode({
        "user_id": user.id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    },
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )

    # REDIRECT TO STREAMLIT
    return redirect(
        f"http://localhost:8501/?token={jwt_token}&email={email}"
    )

if __name__ == '__main__':
    app.run(debug=True)
