
import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Fraud Detection Graph",
    page_icon="🔍",
    layout="wide"
)

st.title("Fraud Detection Network Graph")
st.write("Visual representation of suspicious financial transactions.")

@st.cache_data
def load_data():
    predictions = pd.read_csv("predicted_transactions.csv")

    normal = pd.read_csv(
        "data/transactions/transactions_0_0.csv",
        usecols=["tx_id", "amount"]
    )

    fraud = pd.read_csv(
        "data/fraud/transactions_fraud.csv",
        usecols=["tx_id", "amount"]
    )

    transactions = pd.concat([normal, fraud], ignore_index=True)

    data = predictions.merge(transactions, on="tx_id", how="left")
    return data

try:
    data = load_data()

    data["predicted_fraud"] = pd.to_numeric(
        data["predicted_fraud"], errors="coerce"
    ).fillna(0).astype(int)

    fraud_data = data[data["predicted_fraud"] == 1]
    normal_data = data[data["predicted_fraud"] == 0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(data))
    col2.metric("Fraud Transactions", len(fraud_data))
    col3.metric("Normal Transactions", len(normal_data))

    st.subheader("Transaction Network")

    show_fraud = st.checkbox("Show suspicious transactions", True)
    show_normal = st.checkbox("Show normal transactions", True)

    selected = []

    if show_fraud:
        selected.append(fraud_data.head(100))

    if show_normal:
        selected.append(normal_data.head(50))

    if not selected or all(part.empty for part in selected):
        st.info("Select at least one transaction category.")
    else:
        graph_data = pd.concat(selected, ignore_index=True)

        flagged_accounts = set(
            fraud_data["src_id"].dropna().astype(str)
        ) | set(
            fraud_data["dst_id"].dropna().astype(str)
        )

        net = Network(
            height="650px",
            width="100%",
            directed=True,
            bgcolor="#ffffff",
            font_color="#222222"
        )

        added_nodes = set()

        for _, row in graph_data.iterrows():
            source = str(row["src_id"])
            destination = str(row["dst_id"])

            for account in [source, destination]:
                if account not in added_nodes:
                    is_fraud = account in flagged_accounts

                    net.add_node(
                        account,
                        label=account,
                        color="#e74c3c" if is_fraud else "#3498db",
                        size=25 if is_fraud else 18,
                        title=(
                            f"Account: {account}<br>"
                            f"Status: {'Suspicious' if is_fraud else 'Not flagged'}"
                        )
                    )
                    added_nodes.add(account)

            is_fraud_transaction = row["predicted_fraud"] == 1

            net.add_edge(
                source,
                destination,
                arrows="to",
                color="#e74c3c" if is_fraud_transaction else "#a0a0a0",
                title=(
                    f"Transaction: {row['tx_id']}<br>"
                    f"Amount: {row['amount']}<br>"
                    f"Prediction: {'Fraud' if is_fraud_transaction else 'Normal'}"
                ),
                width=3 if is_fraud_transaction else 1
            )

        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -2500,
              "springLength": 150
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true
          }
        }
        """)

        net.write_html("fraud_network.html", notebook=False)

        with open("fraud_network.html", "r", encoding="utf-8") as file:
            graph_html = file.read()

        components.html(graph_html, height=670, scrolling=True)

        st.markdown("""
        **Graph Legend**

        🔴 Red node / arrow: Suspicious account / transaction

        🔵 Blue node: Account not flagged by the model

        Grey arrow: Transaction predicted as normal
        """)

except FileNotFoundError as error:
    st.error(f"Required project data file not found: {error}")
except Exception as error:
    st.error(f"Something went wrong: {error}")
