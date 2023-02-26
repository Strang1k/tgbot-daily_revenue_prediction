#import logging
import time
import telebot
from telebot import types
import random
import requests, json
from datetime import datetime

# logging.basicConfig( format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
#                     filename = 'bot.log', filemode='a',  force = True ) #без force не создается .log файл

# stream_handler = [h for h in logging.root.handlers if isinstance(h , logging.StreamHandler)][0]
# stream_handler.setLevel(logging.INFO)



#_______________________________________________________________________________________________
#функции предобработки и предскзаания

from pickle import load as pickle_load
from pandas import DataFrame as pd_DataFrame
from pandas import cut as pd_cut
from enum import Enum

#С предсказаниями кбм точно все хорошо с любым форматом 1 и '1'
#knr все ок если формат [0, 1, 5, 2, 0, 'wd']
#lgbmr все ок если формат [1, 2, 3, 4, 5, 'wd]

with open('ohe', 'rb') as f: #one hot encoder
    ohe = pickle_load(f) 

with open('cmbmodel', 'rb') as f: #catboost
    cbm = pickle_load(f)

with open('lgbm', 'rb') as f:
    lgbm = pickle_load(f)  #lgbm

with open('knr', 'rb') as f:
    knr = pickle_load(f)  #kn regressor

with open('rfrblend', 'rb') as f:
    rfrblend = pickle_load(f) #три предикта от предыдущих моделей на вход

with open('cbm7', 'rb') as f:
    cbm7 = pickle_load(f)  #cmb 7avg data

with open('knr7', 'rb') as f:
    knr7 = pickle_load(f)  #knr 7 avg data


# wtl работает с указанием столбца <br/>
# ttb работает просто входом, но на одном примере <br/>
# tocat работает просто с входом на всей выборке <br/>
# ohe работает на всей выборке <br/>


def weekday_to_label(series, weekdayz = 'Ru'): #можно задать и другой enum, исправив функцию
                                                #Удалил основную часть функции т.к. все равно будет подаваться 1 набор

    """
        Список вида ['пн', 'вт', 'ср', 'вс'] превращает в [1, 2, 3, 7]
        Необходимо передать список и вид преобразования или собственную функцию 
        'Ru' - классические ['пн', 'вт', 'ср'] -> [1, 2 , 3]
        'Eng' - ['sun', 'mon', 'tue'] -> [1, 2, 3]
        'Eng1' - ['mon', 'tue', 'wed'] -> [1, 2, 3]
        """

    if weekdayz == 'Ru':
        weekdayz = Enum('weekday numeration', 'пн вт ср чт пт сб вс')

        

    try:
        if str(series) == series:
           
            if len(series) == 2:
                return weekdayz[series.lower()]._value_ #возможно нужно будет поправить
    except:
        print('Неверные данные (неделя) ')
    return series


def month_to_label(series, weekdayz = 'Ru'): #можно задать и другой enum, исправив функцию
                                                #Удалил основную часть функции т.к. все равно будет подаваться 1 набор

    if weekdayz == 'Ru':
        weekdayz = Enum('month numeration', 'январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь')


    try:
        if str(series) == series:
           
            return weekdayz[series.lower()]._value_ 
    except:
        print('Неверные данные (месяц)')
    return series



def temp_to_bins(day_params): #для работоспособности нужен pd.cut(), не работает с списком дней, у меня не получилось
    day_params[3] = int(day_params[3])
    bins = [-50, 0, 15, 25, 50]
    labels = [1, 2, 3, 4]
    if day_params[3] > -50 and day_params[3] < 50:
        day_params[3] = pd_cut([day_params[3]], bins=bins, labels=labels)[0]
        return day_params
    else:
        print('Измените температуру на реалистичную')



def to_category(day_params): 
    for col in day_params.select_dtypes(include=['int64', 'object']):
        day_params[col] = day_params[col].astype('category')

def predobr(smth):
    smth = smth.strip().replace(" ", "")
    y = (smth.split(sep=',')) 
    z = []
    for each in y:
        if each == ',':
            y.remove(each)
    
    for each in y:
        try:
            if int(each) == 0:
                z.append(int(each))
            elif int(each):
                z.append(int(each))
        except ValueError:
            z.append(each)

    return z

def predictpls(message, day_variables):

#Общая предобработка
    day_variables[-1] = weekday_to_label(day_variables[-1])
    day_variables[2]  = month_to_label(day_variables[2])
    day_variables = temp_to_bins(day_variables)
    global models_predictions 
    
    if len(day_variables) == 6 : #без известного 7д авг


        df = pd_DataFrame(columns=['Девочка', 'Мальчик', 'Месяц', 'Температура', 'Дождь', 'День недели'])
        df.loc[0] = day_variables

        x2 = cbm.predict(day_variables) #тут может выдавать предсказания cbm
        #предобработка для knr
        df1 = df.copy()
        df1 = ohe.transform(df1)
        x1 = knr.predict(df1)
        #предобработка для lgbm
        to_category(df)
        x3 = lgbm.predict(df)


        data = {'x1':[x1], 'x2':[x2], 'x3':[x3]}
        
        models_predictions = ('Предсказания отдельных моделей:\n'
            'knr: {k},\ncbm: {c:.2f},\nlgbm: {l:.2f}'.format(k = data['x1'][0][0], c = data['x2'][0], l =data['x3'][0][0]))
        return rfrblend.predict(pd_DataFrame(data=data))

    elif len(day_variables) == 7: #с известным 7д авг
    
    
        df = pd_DataFrame(columns=['Девочка', 'Мальчик', 'Месяц', 'Температура', 'Дождь', 'День недели', '7avg'])
        df.loc[0] = day_variables
        
        x2 = cbm7.predict(day_variables)
        x1 = knr7.predict(df)

        data = {'x1':[x1], 'x2':[x2]}
        models_predictions = ('Предсказания отдельных моделей:\n'
            'knr: {k},\ncbm: {c:.2f}'.format(k = data['x1'][0][0], c = data['x2'][0]))
        return x2*0.8748748748748749+x1*0.12512512512512508

#_______________________________________________________________________________________________
#Рандом дня / 'Сегодня'

def unbracketed(content_line):
    result = []
    for element in content_line:
        if isinstance(element, list):
            result.extend(unbracketed(element))
        else:
            result.append(element)
    return result

def random_day():
    randomday = []
    randomday.append(random.choice([[1, 0], [0, 1]])) #девочка/мальчик
    randomday.append(random.randrange(1, 13)) #Месяц
    randomday.append(random.randrange(-30, 36)) #Темп
    randomday.append(random.randrange(0, 2)) #Дождь/снег
    randomday.append(random.randrange(1, 8)) #День недели
    return unbracketed(randomday)

def weekday_changer(x):
    if x == 0:
        x = 'пн'
    if x == 1:
        x = 'вт'    
    if x == 2:
        x = 'ср'
    if x == 3:
        x = 'чт'
    if x == 4:
        x = 'пт'
    if x == 5:
        x = 'сб'
    if x == 6:
        x = 'вс'
    return x


def today_pred():
    today = []
    today.append(random.choice([[1, 0], [0, 1]])) #девочка/мальчик
    today.append(datetime.today().month) #Месяц
    today.append(round(temp)) #Темп
    today.append(1 if rain or snow > 0 else 0) #Дождь/снег
    today.append(weekday_changer(datetime.today().weekday())) #День недели
    return unbracketed(today)



#_______________________________________________________________________________________________

greeting = ('Привет! В общем, формат такой: \n'
            'Первые две цифры - бинарные, 1 и 0 \n'
            'для Девочки и Мальчика на смене соответственно, \n'
            'Желательно чтобы в сумме они давали 1, \n'
            'Третье число - номер месяца или название целиком, \n'
            'Четвертое число - температура за окном, \n'
            'Пятая - будет ли сегодня дождь (бинарно), \n'
            'Шестая - сокращение или номер дня недели (ср или 3), \n'
            'Седьмая и последная - опционально, если известна средняя выручка за 7 дней\n'
            'В итоге поддерживается формат: \n'
            'Д, М, мес, темп, дождь, Д/н \n'
            '1, 0, Сентябрь, 12, 1,  сб, 7000 или\n'
            '1, 0, 9, 12, 1, 6')

bot = telebot.TeleBot('**********:AA***089FHbRz3***_OHqLIADH09RIQL***')

#telebot.BaseMiddleware

@bot.message_handler(commands=['start'])
def welcome(message):                   #1, 0, 3, 4, 1, ср

    markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1=types.KeyboardButton('Напомни формат')
    item2=types.KeyboardButton('Рандом')
    item3=types.KeyboardButton('Предсказания отдельных моделей')
    item4=types.KeyboardButton('Предсказание на сегодня')
    markup.add(item1)
    markup.add(item2)
    markup.add(item3)
    markup.add(item4)

    bot.send_message(message.chat.id, greeting, reply_markup=markup)


@bot.message_handler(content_types=['text'])
def answer(message):
    if message.text == 'Напомни формат' :
        bot.send_message(message.chat.id, greeting[8:])
    
    elif message.text == 'Рандом' :
        the_random_day = random_day()
        bot.send_message(message.chat.id, 'Рандомно выбрано следующее: \n {}'.format(the_random_day))
        bot.send_message(message.chat.id, 'При таких входных данных \nВыручка будет: {}'.format('%.2f' %predictpls(message, the_random_day)[0]))

    elif message.text == 'Предсказания отдельных моделей':
         bot.send_message(message.chat.id, models_predictions)
    
    elif message.text == 'Предсказание на сегодня':
        fortoday = today_pred()
        bot.send_message(message.chat.id, 'Для "сегодня" получится: \n {}'.format(fortoday))
        bot.send_message(message.chat.id, 'При таких входных данных \nВыручка будет: {}'.format('%.2f' %predictpls(message, fortoday)[0]))


    else:
        bot.send_message(message.chat.id, 'Анализируем...')
        print(message.chat)
        bot.send_message(message.chat.id, message.text)
        bot.send_message(message.chat.id, 'Ща, сек')
        bot.send_message(message.chat.id, 'Кажись, суммарно будет {}'.format(summka(message.text)))
        bot.send_message(message.chat.id, 'Выручка? Выручка будет: {}'.format('%.2f' %predictpls(message, predobr(message.text))[0]))


        do_again(message)      

def summka(list):
    y = (list.split(sep=',')) 
    z = []
    for each in y:
        if each == ',':
            y.remove(each)
    
    for each in y:
        try:
            if int(each):
                z.append(int(each))
        except ValueError:
            pass
    return sum(z)


def do_again(message):
    bot.send_message(message.chat.id, 'Попробуем другие входные данные?')




if __name__ == '__main__':
    while True: 
        try:
            URL = 'https://api.open-meteo.com/v1/forecast?latitude=59.94&longitude=30.31&hourly=temperature_2m,rain,snowfall'
            response = requests.get(URL)
            pogoda = response.json()

            temp = pogoda['hourly']['temperature_2m'][0] #температура
            rain = pogoda['hourly']['rain'][0] #дождь
            snow = pogoda['hourly']['snowfall'][0] #снег
            bot.polling(none_stop=True)
            time.sleep(43200)
        except:
            time.sleep(5)
