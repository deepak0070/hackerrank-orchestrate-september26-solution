import pandas as pd
from pathlib import Path
from data_loader import FinancialDataLoader
from cash_flow import CashFlowEngine

def main():
    print("Initializing Financial Agent Pipeline...")
    
    # 1. Load Data and Extract Images
    loader = FinancialDataLoader()
    print("\nChecking for missing amounts...")
    loader.resolve_missing_amounts()
    
    # 2. Initialize Reasoning Engine
    engine = CashFlowEngine(loader)
    
    # 3. Process all requests
    print("\nEvaluating all financial requests...")
    results = []
    
    requests_df = loader.requests
    total_requests = len(requests_df)
    
    for index, row in requests_df.iterrows():
        req_id = row['request_id']
        print(f"Processing {req_id} ({index + 1}/{total_requests})...")
        
        # Run through our deterministic cash flow engine
        decision = engine.evaluate_request(req_id)
        
        if decision:
            # Enforce schema constraints required by the problem statement
            results.append({
                "request_id": decision["request_id"],
                "amount_safe_to_pay": round(decision["amount_safe_to_pay"], 2),
                "affordability_status": decision["affordability_status"],
                "recommended_payment_method": decision["recommended_payment_method"],
                "payment_plan": decision["payment_plan"],
                "earliest_date_for_full_payment": decision["earliest_date_for_full_payment"],
                "spending_changes_needed": decision["spending_changes_needed"],
                "decision_explanation": decision["decision_explanation"]
            })

    # 4. Save to output.csv
    output_df = pd.DataFrame(results)
    output_path = loader.base_path / "output.csv"
    
    # Ensure columns match exact order requested in problem_statement.md
    columns = [
        "request_id", "amount_safe_to_pay", "affordability_status", 
        "recommended_payment_method", "payment_plan", 
        "earliest_date_for_full_payment", "spending_changes_needed", 
        "decision_explanation"
    ]
    output_df = output_df[columns]
    
    output_df.to_csv(output_path, index=False)
    print(f"\nSuccess! Generated {len(output_df)} predictions saved to {output_path}")

if __name__ == "__main__":
    main()