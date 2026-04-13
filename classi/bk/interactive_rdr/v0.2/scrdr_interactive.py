import pandas as pd
import json
import argparse
import os
from sklearn.metrics import classification_report

class RDRNode:
    def __init__(self, condition=None, conclusion=None):
        self.condition = condition # {'col':.., 'op':.., 'val':..}
        self.conclusion = str(conclusion) if conclusion is not None else None
        self.if_true = None   # Refinement (Exception)
        self.if_false = None  # Alternative

    def evaluate(self, row):
        if not self.condition or not self.condition['col']:
            return True
        col, op, val = self.condition['col'], self.condition['op'], self.condition['val']
        row_val = row[col]
        try:
            if op == '==': return str(row_val) == str(val)
            if op == '<':  return float(row_val) < float(val)
            if op == '>':  return float(row_val) > float(val)
        except:
            return str(row_val) == str(val)
        return False

    def to_dict(self):
        return {
            "condition": self.condition,
            "conclusion": self.conclusion,
            "if_true": self.if_true.to_dict() if self.if_true else None,
            "if_false": self.if_false.to_dict() if self.if_false else None
        }

    @staticmethod
    def from_dict(data):
        if not data: return None
        node = RDRNode(data['condition'], data['conclusion'])
        node.if_true = RDRNode.from_dict(data['if_true'])
        node.if_false = RDRNode.from_dict(data['if_false'])
        return node

class SCRDR_Engine:
    def __init__(self, target, default_conclusion="Unknown"):
        self.target = target
        self.root = RDRNode(conclusion=default_conclusion)

    def classify(self, row):
        curr = self.root
        last_match = self.root
        while curr:
            if curr.evaluate(row):
                last_match = curr
                curr = curr.if_true # Move to exceptions
            else:
                curr = curr.if_false # Move to alternatives
        return last_match

    def add_rule(self, row, last_node):
        print(f"\n[KNOWLEDGE ACQUISITION]")
        print(f"System predicted: '{last_node.conclusion}' but Actual is: '{row[self.target]}'")
        
        # Validation Loop
        while True:
            col = input(f"Enter feature name (from {list(row.index)}): ").strip()
            if col not in row.index:
                print("Invalid column name. Try again.")
                continue
            op = input("Enter operator (==, <, >): ").strip()
            if op not in ['==', '<', '>']:
                print("Invalid operator. Use ==, <, or >")
                continue
            val = input("Enter threshold value: ").strip()
            if not val:
                print("Value cannot be empty.")
                continue
            break
        
        new_node = RDRNode(condition={'col': col, 'op': op, 'val': val}, conclusion=row[self.target])
        
        # If the last node matched, this is an EXCEPTION (if_true)
        if last_node.evaluate(row):
            if last_node.if_true is None:
                last_node.if_true = new_node
            else:
                curr = last_node.if_true
                while curr.if_false: curr = curr.if_false
                curr.if_false = new_node
        else:
            # This is an ALTERNATIVE (if_false)
            curr = last_node
            while curr.if_false: curr = curr.if_false
            curr.if_false = new_node

def main():
    parser = argparse.ArgumentParser(description="Interactive RDR Knowledge Builder")
    parser.add_argument("--input", required=True, help="Dataset CSV file")
    parser.add_argument("--target", required=True, help="Target column to predict")
    parser.add_argument("--tree", default="rdr_model.json", help="Output JSON model file")
    parser.add_argument("--exclude", nargs='*', help="Columns to ignore (IDs, Names)")
    parser.add_argument("--mode", choices=['build', 'test'], default='build', 
                        help="build: Interactive training | test: Run model on data")
    
    args = parser.parse_args()

    # Load Data
    df = pd.read_csv(args.input)
    if args.exclude:
        df = df.drop(columns=[c for c in args.exclude if c in df.columns])

    engine = SCRDR_Engine(args.target)
    
    # Load existing model if it exists
    if os.path.exists(args.tree):
        with open(args.tree, 'r') as f:
            engine.root = RDRNode.from_dict(json.load(f))
        print(f"Loaded existing model: {args.tree}")

    y_true, y_pred = [], []

    print(f"--- Running in {args.mode.upper()} mode ---")
    for idx, row in df.iterrows():
        pred_node = engine.classify(row)
        actual = str(row[args.target])
        predicted = str(pred_node.conclusion)
        
        if args.mode == 'build' and predicted != actual:
            print(f"\nERROR at Row {idx}:")
            print(row.to_dict())
            engine.add_rule(row, pred_node)
            with open(args.tree, 'w') as f:
                json.dump(engine.root.to_dict(), f, indent=2)
            # Re-classify after adding rule
            predicted = str(engine.classify(row).conclusion)

        y_true.append(actual)
        y_pred.append(predicted)

    print("\n--- FINAL PERFORMANCE SUMMARY ---")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()

