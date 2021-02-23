# Heart Failure Prediction


---

**About this dataset**

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

[Here is a link to the dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

### Task
To create a model to assess the likelihood of a death by heart failure event.
This can be used to help hospitals in assessing the severity of patients with cardiovascular diseases.

### Description

Курсовой проект по курсу "Машинное обучение в бизнесе"

Задача: предсказание вероятности Heart failure. Бинарная классификация Используемые признаки: 'age', int 'anaemia', int 'creatinine_phosphokinase', float 'diabetes', int 'ejection_fraction', 
float 'high_blood_pressure', int 'platelets', float 'serum_creatinine', float 'serum_sodium', float 'sex', int (0,1) 'smoking',int (0,1) 'time' int Модель: Random Forest Classifier

Был реализован rest api на базе flask, а также front-end сервис, который умеет принимать от пользователя введеные данные и входить в api.
Дополнительно в данном курсовом проекте был использован Streamlit — бесплатный опенсорсный фреймворк, специально разработанный для специалистов машиннного обучения. Ссылка на описание: https://habr.com/ru/post/473196/



