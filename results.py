import time
import pytz
from predict import predict
from notifications import send_push_message
import firebase
import datetime
import uuid


def output(filename):

    try:

        notifications = firebase.getNotifications()

        saveEventOnDB = firebase.getSaveEvents()

        modelSettings = firebase.getModelSettings()

        start_time = time.time()

        # predict the event
        if modelSettings["accuracy"] == 2:
                results , chartData, duration = predict(filename,False, True, 2)
        elif modelSettings["accuracy"] == 3:
                results , chartData, duration = predict(filename,False, True, 3)
        elif modelSettings["accuracy"] == 4:
                results , chartData, duration = predict(filename,False, True, 4)
        elif modelSettings["accuracy"] == 5:
                results , chartData, duration = predict(filename,False, True, 5)
        else:
                results , chartData, duration = predict(filename)
        
        end_time = time.time()

        if results.__len__() == 0:
            return

        # get the sound event with the highest probability
        max_value = max(chartData.values())
        max_key = [k for k, v in chartData.items() if v == max_value]


        # send notification to the user
        if notifications["on"]:

            if notifications["aggressiveBarks"] and  max_key[0] == "noOfAggressiveChunks":
                    send_push_message("Your dog is aggressively barking!!!")

            if notifications["regularBarks"] and max_key[0] == "noOfNonAggressiveChunks" :
                    send_push_message("Your dog is barking")
            
            if notifications["growling"] and max_key[0] == "noOfGrowlingChunks" :
                    send_push_message("Your dog is aggressive, it's growling!!!")
            
            if notifications["whinning"] and max_key[0] == "noOfWhinningChunks" :
                    send_push_message("Your dog is anxious, it's whinning")


        if saveEventOnDB["on"]:

            # save the event on the database
            local_timezone = pytz.timezone('Asia/Amman')
            current_time = datetime.datetime.now(local_timezone)

            # generate a unique random value
            uniquefilename = str(uuid.uuid4()) + ".wav"

            storage_ref = firebase.save_to_firebase_storage(filename, 'audio/' + uniquefilename)

            event = {
                u'event': "",
                u'timestamp': current_time,
                u'duration': duration,
                u'predictionTime': "{:.2f}".format(end_time - start_time),
                u'audio': storage_ref.name
                
            }

            if saveEventOnDB["aggressiveBarks"] and max_key[0] == "noOfAggressiveChunks" :
                    event["event"] = "aggressive barking"
                    firebase.saveEvent(event)
            
            if saveEventOnDB["regularBarks"] and max_key[0] == "noOfNonAggressiveChunks" :
                    event["event"] = "barking"
                    firebase.saveEvent(event)

            if saveEventOnDB["growling"] and max_key[0] == "noOfGrowlingChunks" :
                    event["event"] = "growling"
                    firebase.saveEvent(event)
            
            if saveEventOnDB["whinning"] and max_key[0] == "noOfWhinningChunks" :
                    event["event"] = "whinning"
                    firebase.saveEvent(event)

            
    except Exception as e:
        print(e)


if __name__ == "__main__":

    path = "D:\\Research Project\\project\my\\22_23-j-14\\test_dataset\\B_LoudBark_AGG_13.wav"
    output(path)