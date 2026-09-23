import csv
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)  # reproducibility matters for an eval you'll report publicly

N_MERCHANTS = 25
N_CUSTOMERS = 2000
DAYS = 14
OUTPUT_PATH = "data/transactions.csv"

FRAUD_RATE = 0.015  # ~1.5% of transactions are part of a fraud pattern -- realistic order of magnitude


def make_merchants(n):
    merchants = []
    for i in range(n):
        merchants.append({
            "merchant_id": f"M{i:03d}",
            "avg_amount": random.uniform(200, 5000),   # INR, rough spread across merchant types
            "amount_std": random.uniform(50, 800),
            "hourly_volume": random.randint(2, 60),     # baseline transactions/hour
            "home_lat": random.uniform(8.0, 28.0),      # rough India lat/lon spread
            "home_lon": random.uniform(72.0, 88.0),
        })
    return merchants


def make_customers(n):
    customers = []
    for i in range(n):
        customers.append({
            "customer_id": f"C{i:05d}",
            "home_device": f"D{random.randint(1, 4000):05d}",
        })
    return customers


def normal_transaction(merchant, customer, ts):
    amount = max(10, random.gauss(merchant["avg_amount"], merchant["amount_std"]))
    return {
        "txn_id": str(uuid.uuid4())[:8],
        "merchant_id": merchant["merchant_id"],
        "customer_id": customer["customer_id"],
        "timestamp": ts.isoformat(),
        "amount": round(amount, 2),
        "device_id": customer["home_device"],
        "geo_lat": round(merchant["home_lat"] + random.gauss(0, 0.05), 4),
        "geo_lon": round(merchant["home_lon"] + random.gauss(0, 0.05), 4),
        "is_fraud": 0,
        "fraud_type": "",
    }


def inject_velocity_spike(merchant, customers, base_ts, rows):
    """Burst of many small transactions from different customers/devices
    hitting the same merchant within a tight window -- classic card-testing."""
    burst_size = random.randint(15, 40)
    window_seconds = random.randint(30, 180)
    for _ in range(burst_size):
        cust = random.choice(customers)
        ts = base_ts + timedelta(seconds=random.uniform(0, window_seconds))
        rows.append({
            "txn_id": str(uuid.uuid4())[:8],
            "merchant_id": merchant["merchant_id"],
            "customer_id": cust["customer_id"],
            "timestamp": ts.isoformat(),
            "amount": round(random.uniform(1, 50), 2),  # small "testing" amounts
            "device_id": f"D{random.randint(4001, 9000):05d}",  # unfamiliar device
            "geo_lat": round(merchant["home_lat"] + random.gauss(0, 0.05), 4),
            "geo_lon": round(merchant["home_lon"] + random.gauss(0, 0.05), 4),
            "is_fraud": 1,
            "fraud_type": "velocity_spike",
        })


def inject_amount_anomaly(merchant, customer, base_ts, rows):
    """One transaction far outside the merchant's normal amount range."""
    multiplier = random.uniform(8, 25)
    amount = merchant["avg_amount"] * multiplier
    rows.append({
        "txn_id": str(uuid.uuid4())[:8],
        "merchant_id": merchant["merchant_id"],
        "customer_id": customer["customer_id"],
        "timestamp": base_ts.isoformat(),
        "amount": round(amount, 2),
        "device_id": customer["home_device"],
        "geo_lat": round(merchant["home_lat"] + random.gauss(0, 0.05), 4),
        "geo_lon": round(merchant["home_lon"] + random.gauss(0, 0.05), 4),
        "is_fraud": 1,
        "fraud_type": "amount_anomaly",
    })


def inject_geo_device_anomaly(merchant, customer, base_ts, rows):
    """Transaction from a location/device far from both merchant and customer norms."""
    rows.append({
        "txn_id": str(uuid.uuid4())[:8],
        "merchant_id": merchant["merchant_id"],
        "customer_id": customer["customer_id"],
        "timestamp": base_ts.isoformat(),
        "amount": round(random.gauss(merchant["avg_amount"], merchant["amount_std"]), 2),
        "device_id": f"D{random.randint(4001, 9000):05d}",
        "geo_lat": round(random.uniform(8.0, 28.0), 4),   # random far-away location
        "geo_lon": round(random.uniform(72.0, 88.0), 4),
        "is_fraud": 1,
        "fraud_type": "geo_device_anomaly",
    })


def generate():
    merchants = make_merchants(N_MERCHANTS)
    customers = make_customers(N_CUSTOMERS)
    start = datetime(2026, 8, 1)
    rows = []

    for day in range(DAYS):
        for merchant in merchants:
            n_txns_today = int(random.gauss(merchant["hourly_volume"] * 24, merchant["hourly_volume"] * 3))
            n_txns_today = max(5, n_txns_today)
            for _ in range(n_txns_today):
                ts = start + timedelta(days=day, seconds=random.uniform(0, 86400))
                cust = random.choice(customers)
                rows.append(normal_transaction(merchant, cust, ts))

            # decide if this merchant gets hit by a fraud pattern today
            if random.random() < FRAUD_RATE * 20:  # tuned so we get a reasonable number of incidents
                pattern = random.choice(["velocity_spike", "amount_anomaly", "geo_device_anomaly"])
                base_ts = start + timedelta(days=day, seconds=random.uniform(0, 86400))
                if pattern == "velocity_spike":
                    inject_velocity_spike(merchant, customers, base_ts, rows)
                elif pattern == "amount_anomaly":
                    cust = random.choice(customers)
                    inject_amount_anomaly(merchant, cust, base_ts, rows)
                else:
                    cust = random.choice(customers)
                    inject_geo_device_anomaly(merchant, cust, base_ts, rows)

    rows.sort(key=lambda r: r["timestamp"])

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_fraud = sum(r["is_fraud"] for r in rows)
    print(f"Generated {len(rows)} transactions, {n_fraud} fraudulent ({n_fraud/len(rows)*100:.2f}%)")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()