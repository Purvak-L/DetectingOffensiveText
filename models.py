from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import render_template
from flask import Flask, flash, redirect, url_for
from config import Config
from forms import LoginForm


app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    handle = db.Column(db.String(120), index=True, unique=True)

    def __repr__(self):
        return '<User {}>'.format(self.username)    
