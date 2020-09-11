import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIASSEVXQOYMOTOC4MK",
                        aws_secret_access_key="ZsGkLBVxynoyLixuN/ZbhzdT664mdWIrkcGUcQoV",
                        aws_session_token="FwoGZXIvYXdzEIP//////////wEaDAj/v7NoZxNvi87FxSLNAdwev1URXuoIyKAiD1GvH5jUVJDZysi0eDVY3u2tBTIa5MDD/CAUssuxU3Sb81YnqWWya4DqtYGqkCIcBOqD82bJZBhGsgdQg4OLdosltcGBQu00FvzzjrYPM22SdxLn3KXuZQEzPPYVyQx8wvlJTppJcgAJk6VxOGuKyxUWzoBTQR2ku1df4rx9B6PLOee1RDHggxJ/fmLrha20dXUfYe8gAmbgR2MCpT3wOm/3o2XA4oc06aglcY24RJdcCHRdi1o3MPtIJU4h5/1NWMUoy+7u+gUyLb46ymOO2EA/86oLl24DPBLyiPQdAznWb+CIEr8xllns++cQ3K+nmnGqfeTmOA==",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:176407675824:project/Mask-Detection/version/Mask-Detection.2020-09-11T16.44.27/1599822870948',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://xvbzm9r8zi.execute-api.us-east-1.amazonaws.com/deploymaskapi?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
