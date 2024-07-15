import plotly.express as px
import pandas

df = pandas.read_csv(rf'Project\other_models_1.csv')
fig = px.bar(df, x="Model", y="Average", color="Model")

# df = pandas.read_csv(rf'C:\Users\redel\Desktop\Code\SHINE\Project\og_model.csv')
# fig = px.scatter(df, x="Learning rate", y="Iterations", size="Avg Accuracy")

fig.show()
