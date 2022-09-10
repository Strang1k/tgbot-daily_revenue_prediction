import logging
import time
import telebot
from telebot import types
import random


logging.basicConfig(filemode='bot.log', level=logging.DEBUG)
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
        print('Неверные данные')
    return series


def month_to_label(series, weekdayz = 'Ru'): #можно задать и другой enum, исправив функцию
                                                #Удалил основную часть функции т.к. все равно будет подаваться 1 набор

    if weekdayz == 'Ru':
        weekdayz = Enum('month numeration', 'январь февраль март апрель май инюнь июль август сентябрь октябрь ноябрь декабрь')


    try:
        if str(series) == series:
           
            return weekdayz[series.lower()]._value_ 
    except:
        print('Неверные данные')
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

def predictpls(day_variables):

#Общая предобработка
    day_variables[-1] = weekday_to_label(day_variables[-1])
    day_variables[2]  = month_to_label(day_variables[2])
    day_variables = temp_to_bins(day_variables)
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
    return rfrblend.predict(pd_DataFrame(data=data))

#_______________________________________________________________________________________________
#Рандом дня

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
    randomday.append(random.choice([[1, 0], [0, 1]]))
    randomday.append(random.randrange(1, 13))
    randomday.append(random.randrange(-30, 36))
    randomday.append(random.randrange(0, 2))
    randomday.append(random.randrange(1, 8))
    return unbracketed(randomday)

#_______________________________________________________________________________________________

greeting = ('Привет! В общем, формат такой: \n'
            'Первые две цифры - бинарные, 1 и 0 \n'
            'для Девочки и Мальчика на смене соответственно, \n'
            'Желательно чтобы в сумме они давали 1, \n'
            'Третье число - номер месяца или название целиком, \n'
            'Четвертое число - температура за окном, \n'
            'Пятая - будет ли сегодня дождь (бинарно), \n'
            'Шестая и последняя - сокращение или номер дня недели (ср или 3). \n'
            'В итоге поддерживается формат: \n'
            'Д, М, мес, темп, дождь, Д/н \n'
            '1, 0, Сентябрь, 12, 1,  сб  или\n'
            '1, 0, 9, 12, 1, 6')

bot = telebot.TeleBot('**********:AA***089FHbRz3***_OHqLIADH09RIQL***')


@bot.message_handler(commands=['start'])
def welcome(message):                   #1, 0, 3, 4, 1, ср

    markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1=types.KeyboardButton("Напомни формат")
    item2=types.KeyboardButton("Рандом")
    markup.add(item1)
    markup.add(item2)

    bot.send_message(message.chat.id, greeting, reply_markup=markup
                                      )


@bot.message_handler(content_types=['text'])
def answer(message):
    if message.text == 'Напомни формат' :
        bot.send_message(message.chat.id, greeting[8:])
    
    elif message.text == 'Рандом' :
        the_random_day = random_day()
        bot.send_message(message.chat.id, 'Рандомно выбрано следующее: \n {}'.format(the_random_day))
        bot.send_message(message.chat.id, 'При таких входных данных \nВыручка будет: {}'.format('%.2f' %predictpls(the_random_day)[0]))


    else:
        bot.send_message(message.chat.id, 'Анализируем...')
        # passenger_data = message.text.split()
        # passenger_data.insert(0, 0)
        # passenger_data.insert(9, ',')
        # passenger_data[2] = '"',passenger_data[2],'"'

        #answer = predict_d_r(list(message)) #passenger_data
        print(message)
        bot.send_message(message.chat.id, message.text)
        bot.send_message(message.chat.id, 'Ща, сек')
        bot.send_message(message.chat.id, 'Формат {}'.format(type(message.text)))
        bot.send_message(message.chat.id, 'Кажись, суммарно будет {}'.format(summka(message.text)))
        bot.send_message(message.chat.id, 'Выручка? Выручка будет: {}'.format('%.2f' %predictpls(predobr(message.text))[0]))


        # if answer:
        #     bot.send_message(message.chat.id, 'Исходя из наблюдений, выручка должна быть такой:')
        #     answer = predict_d_r(list(message))
        
        
        # bot.send_message(message.chat.id, 'Анализируем...')


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
            bot.polling(none_stop=True)
        except:
            time.sleep(5)