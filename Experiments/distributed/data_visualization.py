import datetime
import pytz

time = datetime.datetime.now(pytz.timezone('Europe/Berlin'))
print(str(time))