import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
from pydub import AudioSegment

cred = credentials.Certificate("./key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'woofyk-a6755.appspot.com'
})

db = firestore.client()

bucket = storage.bucket()

UID = "001"


def getExpoToken() :
    doc_ref =  db.collection(u'users').document(UID)
    doc =  doc_ref.get()
    return doc.to_dict()['token']



def saveEvent(event) :
    doc_ref =  db.collection(u'events').document(UID)
    doc =  doc_ref.get()
    events = doc.to_dict()['events']
    events.append(event)
    doc_ref.update({u'events': events})

def getNotifications() :
    doc_ref =  db.collection(u'users').document(UID)
    doc =  doc_ref.get()
    return doc.to_dict()['notifications']

def getSaveEvents() :
    doc_ref =  db.collection(u'users').document(UID)
    doc =  doc_ref.get()
    return doc.to_dict()['saveEvent']

def getModelSettings() :
    doc_ref =  db.collection(u'users').document(UID)
    doc =  doc_ref.get()
    return doc.to_dict()['model']


def save_to_firebase_storage(file_path, destination_path):
    """
    Saves a file to Firebase Storage.
    :param file_path: The local path of the file to save.
    :param destination_path: The path where the file should be saved in Firebase Storage.
    :return: The storage reference of the saved file.
    """
    # Compress audio
    sound = AudioSegment.from_file(file_path)
    sound.export(file_path, format="mp3", parameters=["-q:a", "10"])

    # Upload compressed audio file to Firebase Storage
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(file_path)

    return blob