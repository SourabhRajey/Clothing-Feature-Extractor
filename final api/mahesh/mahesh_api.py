import tornado.ioloop
import tornado.web
import json
import requests
import psycopg2
from psycopg2 import pool


#database pool connection variable
postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(1, 20,user = "mahi",
            password = "1234567890",
            host = "localhost",
            port = "5432",
            database = "firstcrydata")

#==============================================================================================
#==============================================================================================


def getData(typeofcloth,color,pattern,gender):
    try:
        ps_connection  = postgreSQL_pool.getconn()
        if(ps_connection):
            ps_cursor = ps_connection.cursor()
            query = """select fcid from temp where type in (%s) and color in (%s) and gender in (%s) and pattern in (%s) order by random() limit 30"""
            param = (typeofcloth,color,gender,pattern)
            ps_cursor.execute(query,param)
            records = ps_cursor.fetchall()
            ps_cursor.close()
            postgreSQL_pool.putconn(ps_connection)
            return records
    except (Exception, psycopg2.DatabaseError) as error :
        print ("Error in retrieve query", error)
    finally:
        if(postgreSQL_pool):
            postgreSQL_pool.closeall


def insertRequestData(tid,timestamp,imgpath,typeofcloth,color,pattern,necktype):
    try:
        ps_connection  = postgreSQL_pool.getconn()
        if(ps_connection):
            ps_cursor = ps_connection.cursor()
            query="""INSERT INTO requests values(%s,%s,%s,%s,%s,%s,%s)"""
            data = (tid,timestamp,imgpath,typeofcloth,color,pattern,necktype)
            ps_cursor.execute(query,data)
            ps_connection.commit()
            ps_cursor.close()
            postgreSQL_pool.putconn(ps_connection)
    except (Exception, psycopg2.DatabaseError) as error :
        print ("Error in insert query", error)
    finally:
        if(postgreSQL_pool):
            postgreSQL_pool.closeall


def insertData(fcid,typeofcloth,gender,sleeves,necktype):
    try:
        ps_connection  = postgreSQL_pool.getconn()
        if(ps_connection):
            ps_cursor = ps_connection.cursor()
            query="""INSERT INTO data values(%s,%s,%s,%s,%s)"""
            data = (fcid,typeofcloth,gender,sleeves,necktype)
            ps_cursor.execute(query,data)
            ps_connection.commit()
            ps_cursor.close()
            postgreSQL_pool.putconn(ps_connection)
    except (Exception, psycopg2.DatabaseError) as error :
        print ("Error in insert query", error)
    finally:
        if(postgreSQL_pool):
            postgreSQL_pool.closeall


def getfcid(typeofcloth,gender,sleevetype,necktype):
    try:
        ps_connection  = postgreSQL_pool.getconn()
        if(ps_connection):
            ps_cursor = ps_connection.cursor()
            query = """select fcid from vitondata where type in (%s) and gender in (%s) and sleeves in (%s) and necktype in (%s)"""
            param = (typeofcloth,gender,sleevetype,necktype)
            ps_cursor.execute(query,param)
            records = ps_cursor.fetchall()
            ps_cursor.close()
            postgreSQL_pool.putconn(ps_connection)
            return records
    except (Exception, psycopg2.DatabaseError) as error :
        print ("Error in retrieve query", error)
    finally:
        if(postgreSQL_pool):
            postgreSQL_pool.closeall
#============================================================================================
#============================================================================================


class GetDataValue(tornado.web.RequestHandler):
    def post(self):
        print("Request Came")
        value1 = self.get_argument('type')
        value2 = self.get_argument('color')
        value3 = self.get_argument('pattern')
        value4 = self.get_argument('gender')
        #value5 = self.get_argument('necktype')
        print(value1, value2, value3, value4)
        fetcheddata = getData(value1,value2,value3,value4)
        if len(fetcheddata)>0:
            reply = "{"
            i=1
            for row in fetcheddata:
                reply += '"fcid'+str(i)+'":"'+row[0]+'",'
                i +=1
            reply = reply[:-1]
            reply += "}"
            res = json.loads(reply)
            print(res)
            self.write(res)
        else:
            self.write("No data found")


class InsertRequestData(tornado.web.RequestHandler):
    def post(self):
        tid = self.get_argument('tid')
        timestamp = self.get_argument('timestamp')
        imgpath = self.get_argument('imgpath')
        typeofcloth = self.get_argument("type")
        color = self.get_argument("color")
        pattern = self.get_argument("pattern")
        necktype = self.get_argument('necktype')
        insertRequestData(tid,timestamp,imgpath,typeofcloth,color,pattern,necktype)


class DumpData(tornado.web.RequestHandler):
	def post(self):
		print("request came")
		fcid = self.get_argument('fcid')
		typeofcloth = self.get_argument('type')
		gender = self.get_argument('gender')
		sleeves = self.get_argument('sleeve')
		necktype = self.get_argument('necktype')
		insertData(fcid,typeofcloth,gender,sleeves,necktype)



class GetFcid(tornado.web.RequestHandler):
    def post(self):
        print("Request Came")
        typeofcloth = self.get_argument('type')
        gender = self.get_argument('gender')
        sleevetype = self.get_argument('sleevetype')
        necktype = self.get_argument('necktype')
        fetcheddata = getfcid(typeofcloth,gender,sleevetype,necktype)
        if len(fetcheddata)>0:
            reply = "{"
            i=1
            for row in fetcheddata:
                reply += '"fcid'+str(i)+'":"'+row[0]+'",'
                i +=1
            reply = reply[:-1]
            reply += "}"
            res = json.loads(reply)
            self.write(res)
        else:
            self.write("No data found")


#==============================================================================================
#==============================================================================================


def make_app():
    return tornado.web.Application([
        (r"/search", GetDataValue),
        (r"/dumprequest", InsertRequestData),
        (r"/dumpdata", DumpData),
        (r"/getfcids", GetFcid),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()


#==============================================================================================


'''
class GetDataValue(tornado.web.RequestHandler):
	def post(self):
		typeofcloth = self.get_argument('type')
		color = self.get_argument('color')
		pattern = self.get_argument('pattern')
		gender = self.get_argument('gender')
		necktype = self.get_argument('necktype')
		#payloadstr = self.get_argument('payloadstr')
		#sleeves = self.get_argument('sleeves')
		sleeves = ""
		print("request came ",typeofcloth, color, pattern, gender, necktype, sleeves)
		r = requests.get("https://www.firstcry.com/svcs/ProductFilter.svc/GetSubcategoryWiseFilterProducts?"+
			"PageNo=1&PageSize=20&SortExpression=Popularity&SubCatId="+typeofcloth+
			"&BrandId=&Price=&Age=&Color="+color+"&OptionalFilter=&OutOfStock=&Type1="+
			"&Type2=&Type3="+necktype+"&Type4="+sleeves+"&Type5=&Type6="+pattern+"&Type7="+
			"&Type8=&Type9=&Type10=&combo=&discount=&searchwithincat=&ProductidQstr=&searchrank=&"+
			"pmonths=&cgen=&PriceQstr=&DiscountQstr=&sorting=&rating=&offer=&CatId=6&skills=&"+
			"material=&measurement=&gender="+gender+"&exclude=&p=&premium=")
		str = json.loads(r.text)
		str1 = json.loads(str["ProductResponse"])
		res = str1["Products"]
		arr = []
		msg =''
		for pid in res:
			arr.append(pid["PId"])
		if(len(arr)==0):
			msg = "Sorry, No match found" 
		else:
			msg = "Recommended clothes"
		self.render("response.html",items=arr,msg = msg)
'''