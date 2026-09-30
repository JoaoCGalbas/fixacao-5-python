import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.DataFrame({
    "id_cliente": np.random.choice(["C100", "C101", "C102", "C103", "C104"], size=1000),
    "data_transacao": pd.date_range(start="2026-09-01", periods=1000, freq="h"),
    "valor": np.random.choice(
        [np.nan, 150.0, 3000.0, 7500.0, 12000.0, 50.0],
        size=1000,
        p=[0.05, 0.40, 0.30, 0.15, 0.05, 0.05],
    ),
    "estado_cliente": np.random.choice(["SP", "RJ", "MG", "RS"], size=1000),
    "origem": np.random.choice(["web", "mobile_app", "atm"], size=1000),
})


df = pd.concat([df, df.iloc[:15]], ignore_index=True)

df.to_csv("transacoes.csv", index=False, encoding="latin1")
print("transacoes.csv gerado!")
# 1. Ingestão e limpeza
df = pd.read_csv("transacoes.csv", encoding="latin1")
df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
df["valor"] = df["valor"].fillna(df.groupby("estado_cliente")["valor"].transform("median"))
df["plataforma"] = "Mobile"


df["data_transacao"] = (
    pd.to_datetime(df["data_transacao"]).astype("datetime64[ns]").dt.tz_localize("America/Sao_Paulo")
)
df["dia_semana"] = df["data_transacao"].dt.day_name()
df["mes"] = df["data_transacao"].dt.month

n_antes = len(df)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"Duplicatas removidas: {n_antes - len(df)}")


mask = (
    (df["mes"] == 9)
    & ((df["estado_cliente"] == "SP") | (df["estado_cliente"] == "RJ"))
    & (df["valor"] > 5000)
)
df_filtrado = df[mask]
print(f"Filtro (set, SP/RJ, > 5k): {len(df_filtrado)} transações")


risco_dict = {"C100": "Baixo", "C101": "Alto", "C102": "Médio", "C103": "Baixo", "C104": "Alto"}
df["nivel_risco"] = df["id_cliente"].map(risco_dict)

pivot = df.pivot_table(
    index="mes", columns="nivel_risco", values="valor",
    aggfunc="sum", margins=True, margins_name="Total",
)
print(pivot)


def zscore_por_estado(data: pd.DataFrame) -> pd.Series:
    g = data.groupby("estado_cliente")["valor"]
    return (data["valor"] - g.transform("mean")) / g.transform("std", ddof=0).replace(0, np.nan)

df["zscore"] = zscore_por_estado(df)
df_anomalias = df[df["zscore"] > 2.5].copy()
print(f"Anomalias (Z > 2.5): {len(df_anomalias)}")


cot = pd.read_csv("cotacoes.csv", encoding="utf-8", parse_dates=["data"]).sort_values("data")
cot["data"] = cot["data"].astype("datetime64[ns]")

cot["cotacao_usd"] = cot["cotacao_usd"].interpolate()

df["data"] = df["data_transacao"].dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")  #
df = df.sort_values("data")
df = pd.merge_asof(df, cot[["data", "cotacao_usd"]], on="data", direction="backward")


diario = df.set_index("data_transacao")["valor"].resample("D").sum()
media_movel = diario.rolling(7).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(diario.index, diario.values, label="Total diário", alpha=0.6)
ax.plot(media_movel.index, media_movel.values, label="Média móvel (7 dias)", linewidth=2.5)
ax.set_ylim(bottom=0)
ax.set_title("Valor total de transações por dia")
ax.set_xlabel("Data")
ax.set_ylabel("Valor (R$)")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("transacoes_diarias.png", dpi=150)
plt.show()