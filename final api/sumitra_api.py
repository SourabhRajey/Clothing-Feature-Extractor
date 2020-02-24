import tornado.ioloop
import tornado.web
import json
import requests

typesdict = {
    "Tee" : "26",
    "Top" : "26",
    "Tank": "26",
    "Sweater" : "219",
    "Button-Down" : "220",
    "Shirt" : "220",
    "Blouse" : "220",
    "Flannel" : "220",
    "Cardigan" : "220",
    "Jacket" : "267",
    "Anorak" : "267",
    "Parka" : "267",
    "Bomber" : "267",
    "Henley" : "26",
    "Skirt" : "221",
    "Shorts" : "221",
    "Jeans" : "221",
    "Peacoat" : "267",
    "Leggings" : "269",
    "Chinos" : "221",
    "Dress" : "243,168"
}

genderdict = {
    "Tee" : "male",
    'Top' : 'female',
    'Tank' : 'male,female',
    'Henley' : 'male',
    'Sweater' : 'male,female',
    'Button-Down' : 'male',
    'Shirt' : 'male',
    'Blouse' : "male",
    'Cardigan' : "male",
    'Jacket' : 'male,female',
    'Skirt' : 'female',
    'Chinos' : 'male,female',
    'Peacoat' : 'male',
    'Shorts' : 'male,female',
    'Jeans' : 'male,female',
    'Flannel' : 'male',
    'Leggings' : 'female',
    'Dress' : 'male,female',
    'Anorak' : 'male,female',
    'Parka' : 'male,female',
    'Bomber' : 'male,female',
}

colordict = { 
    "black" : "2",
    "blue" : "3",
    "red" : "19",
    "brown" : "4",
    "green" : "7",
    "gray" : "8",
    "maroon" : "9",
    "pink" : "16",
    "purple" : "18",
    "white" : ['11','21','20'],
    "yellow" : "22",
    "Golden" : "22",
    "mustard" : "22"
}

patterndict = {
    "Solid" : "10410",
    "Graphics" : '10415,10411',
    "Floral" : '10417,10412',
    "Striped" : "10413",
    "Spotted" : "10630",
    "Plaid" : "10414"
}

necktypedict = {
    "round_neck" : "384",
    "polo" : "385",
    "v_neck" : "386"
}
'''
sleevelengthdict = {
    "half-sleeves" : "10405",
    "full-sleeves" : "10404",
    "no-sleeves" : "10408",
}
'''
def fun(typeofcloth,color,pattern,necktype,payloadstring):
    value1 = typesdict[typeofcloth]
    value2 = colordict[color]
    value3 = patterndict[pattern]
    value4 = genderdict[typeofcloth]
    if necktype != "None":
        value5 = necktypedict[necktype]
    else:
        value5=""
    payloadstring +=", gender : "+value4 
    payload = {'type' : value1, 'color':value2, 'pattern' : value3, 'gender' : value4, 'necktype' : value5}
    r = requests.post('http://10.0.0.218:8888/search',params=payload)
    return r


class Process(tornado.web.RequestHandler):
    def post(self):
        typeofcloth = self.get_argument('type')
        color = self.get_argument('color')
        pattern = self.get_argument('pattern')
        necktype = self.get_argument('necktype')
        print("request - ",typeofcloth,color,pattern,necktype)
        payloadstring = "type : "+typeofcloth+", color : "+color+", pattern : "+pattern+", necktype : "+necktype
        r = fun(typeofcloth,color,pattern,necktype,payloadstring)
        #res = json.loads(r.text)
        #self.render("response.html",res=res)Sweater
        self.write(r.text)


application = tornado.web.Application([
    (r"/process", Process),
    ])
 
if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()