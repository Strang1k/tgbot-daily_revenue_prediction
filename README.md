# daily_revenue_prediction
The try to figure out what mostly affects daily income from data I collected

Данные для модели используются с разрешения управляющей <br/>

.viruchka визуализация, выбросы, подбор параметров <br/>

.viruchka_short отдельный файл с загрузкой обученных моделей (только на треин дф), <br/>
и помещение их в функцию, где по предсказаниям 3 моделей случайный лес выдает итоговое предсказание <br/>

Показался интересным момент с shrink() в catboost и разница в MSE (можно найти по shrink) <br/>
Доволен написанными функциями week_to_label и select_weights <br/>

.tgbot Добавлен тг бот отдельной папкой, упакован в докер и перенесен на сервер <br/>
Бот доступен по ссылке https://t.me/daily_rev_pred_bot


