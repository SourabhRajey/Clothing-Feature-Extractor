import tornado.ioloop
import tornado.web
import json
import re
import requests
import datetime
import Attribute_prediction

def senddata(imgpath):
    temp = datetime.datetime.now()
    stamp = str(temp)
    tid = re.sub(':|-|\.| ', "", stamp)
    data = Attribute_prediction.run_example(imgpath)
    print(data)
    payload1 = {'type':data["type"],'color':data["color"],'pattern':data["pattern"],'necktype':data["neckline"]}
    r1 = requests.post('http://10.0.1.13:8888/process',params=payload1)
    payload2 = {'tid':tid,'tname':'requests','timestamp':stamp,'imgpath':imgpath,
    'type':data["type"],'color':data["color"],'pattern':data["pattern"],'necktype' :data["neckline"]}
    requests.post('http://10.0.0.218:8888/dumprequest',params=payload2)
    return r1


class SendImagePath(tornado.web.RequestHandler):
    def post(self):
        imgpath = self.get_argument('imgpath')
        r = senddata(imgpath)
        self.write(r.text)


def make_app():
    return tornado.web.Application([
        (r"/sendimage",SendImagePath),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
