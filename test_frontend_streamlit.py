import json
import asyncio
import streamlit as st
import pandas as pd
import aiohttp
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

API_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
intro = '''
[Интро]
Это второй
А
That's a Krishtall
Ау-у, YEEI, а

[Припев]
52 (Алло)
Да здравствует Санкт-Петербург (А), и это город наш (YEEI)
Я каждый свой новый куплет валю как никогда (YEEI, а)
Альбом, он чисто мой, никому его не продам (Он мой)
Не думаю о том (YEEI), как хорошо было вчера (А-а; мне пох)

[Куплет]
Меняю города (А)
Представляю район — у меня есть репертуар (YEEI, 2-3)
Никогда не просил, но всегда где-то доставал (Где?)
Чем больше денег (А), тем больше мне нравится Москва (А)
Но в Питере душа (YEEI), в Питере семья (YEEI)
В Питере братва (А, а), там знают наши имена (52)
+7(952)8-1-2 (Алло)
Это второй альбом (А), вторая глава (Второй)
Не думал, не гадал, всё, что я делал, — рэповал (Всегда)
Андеграунд — это не броуки в протёртых штанах (Пошёл на хуй)
Нужно прожить мою жизнь, чтоб так же, как я, слагать (Ага)
Нужно мой рэп услышать (YEEI), чтоб точно его понять
See upcoming rap shows
Get tickets for your favorite artists
You might also like
Первомай (May Day)
Валентин Стрыкало (Valentin Strikalo)
Ржавый (Rusty)
Кишлак (Kishlak)
GoodDayFlopTray
DJ Stonik1917
[Припев]
52 (Алло)
Да здравствует Санкт-Петербург (А), и это город наш (YEEI)
Я каждый свой новый куплет валю как никогда (YEEI, а)
Альбом, он чисто мой, никому его не продам (Он мой)
Не думаю о том (YEEI), как хорошо было вчера (Ага)

[Аутро]
Да здравствует 52
Да здравствует Петербург, да здравствует 52
Да здравствует Петербург, да здравствует 52 (Ау; YEEI, а)
Да здравствует 52 (Ау), YEEI, long live (Это второй)
'''
st.title(intro)

@st.cache_data
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data = data.sort_values(by=["city", "timestamp"]).reset_index(drop=True)
    
    data["rolling_mean"] = None
    for city in data["city"].unique():
        data.loc[data.city == city, "rolling_mean"] = data.loc[data.city == city, "temperature"].rolling(window=30).mean()
    
    stats = data.groupby(by=["city", "season"], as_index=False).agg({"temperature": ["mean", "std"]})
    stats.columns = ["_".join(col).strip() for col in stats.columns]
    stats = stats.rename(columns={'city_': 'city', 'season_': 'season'})
    print(stats.columns)
    data = data.merge(stats, on=["city", "season"], how="left")
    data["anomaly"] = ((data["temperature"] < data.temperature_mean - 2 * data.temperature_std) | 
                       (data["temperature"] > data.temperature_mean + 2 * data.temperature_std)).astype(int)
    
    return data, stats

async def fetch_weather_data(city, api_key):
    url = API_URL.format(city=city, API_KEY=api_key)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return json.loads(await response.text())

def plot_city_temperature_map(city, temperature):
    url_natural_earth = 'https://www.naturalearthdata.com/downloads/110m-cultural-vectors/naturalearth_lowres'
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    city_data = world[world.name == city]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    city_data.plot(column='temperature', ax=ax, legend=True, cax=cax, cmap='coolwarm', 
                   legend_kwds={'label': "Temperature (°C)", 'orientation': "vertical"})
    ax.set_title(f'Temperature in {city}')
    return fig

def main():
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload a CSV file.")
        return
    data, stats = load_and_process_data(uploaded_file)
    selected_city = st.selectbox("Choose a city:", data["city"].unique())
    api_key = st.text_input("Enter OpenWeather API Key:")
    
    if api_key:
        weather_data = asyncio.run(fetch_weather_data(selected_city, api_key))
        if "main" not in weather_data:
            st.error(weather_data["message"])
            return 
        current_temp = weather_data["main"]["temp"]
        st.success(f"Current temperature in {selected_city}: **{current_temp} °C**.")
        winter_mean, winter_std = stats.loc[(stats.city == selected_city) & (stats.season == "winter"), 
                                             ["temperature_mean", "temperature_std"]].values[0]
        
        anomaly_flag = "not detected"
        if (current_temp > winter_mean + 2 * winter_std) or (current_temp < winter_mean - 2 * winter_std):
            anomaly_flag = "detected"
        st.markdown(f"Normal temperature bounds: **({winter_mean - 2 * winter_std:.2f}°C, {winter_mean + 2 * winter_std:.2f}°C)**. Anomaly status: {anomaly_flag}.")
        st.markdown("#")
        st.markdown(f"##### Temperature Distribution in {selected_city}.")
        fig, ax = plt.subplots()
        sns.boxplot(data=data[data.city == selected_city], x="season", y="temperature", ax=ax)
        st.pyplot(fig)
        st.markdown("#")
        st.markdown(f"##### Historical Temperature in {selected_city}.")
        fig, ax = plt.subplots(figsize=(15, 8))
        city_data = data[data.city == selected_city]
        city_data["timestamp"] = pd.to_datetime(city_data["timestamp"])
        anomalies = city_data[city_data.anomaly == 1]

        ax.plot(city_data["timestamp"], city_data["temperature"], label="Temperature")
        ax.plot(city_data["timestamp"], city_data["temperature_mean"] - 2 * city_data["temperature_std"], linestyle='dashed', label="Lower Bound")
        ax.plot(city_data["timestamp"], city_data["temperature_mean"] + 2 * city_data["temperature_std"], linestyle='dashed', label="Upper Bound")
        ax.scatter(anomalies["timestamp"], anomalies["temperature"], color="red", marker="*", label="Anomaly")
        ax.legend()
        
        st.pyplot(fig)
        st.markdown("#")
        st.markdown(f"##### Seasonal Statistics for {selected_city}.")
        st.dataframe(stats[stats.city == selected_city].reset_index(drop=True))
        st.markdown("#")
        st.markdown(f"##### Temperature Map for {selected_city}.")
        map_fig = plot_city_temperature_map(selected_city, current_temp)
        st.pyplot(map_fig)

if __name__ == "__main__":
    st.video('https://www.youtube.com/watch?v=4OTDBRO38O0', loop=True)
    main()
