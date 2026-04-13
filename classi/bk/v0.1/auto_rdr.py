import pandas as pd
import json
import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class RDRNode:
    def __init__(self, condition=None, conclusion=None, cornerstone=None):
        """
        SCRDR Node structure:
        - condition: {'col': name, 'op': '>', 'val': threshold}
        - cornerstone: The specific data row that justifies this rule.
        """
        self.condition = condition
        self.conclusion = conclusion
        self.cornerstone = cornerstone
        self.if_true = None  # Exception branch
        self.if_false = None # Alternative/Next branch

    def evaluate(self, row):
        if self.condition is None: return True
        col, op, val = self.condition['col'], self.condition['op'], self.condition['val']
        try:
            target = float(row[col])
            threshold = float(val)
            if op == '>': return target > threshold
            if op == '<=': return target <= threshold
        except:
            # Fallback for string comparison
            return str(row[col]) == str(val)
        return False

    def to_dict(self):
        """Returns a nested dictionary for JSON tracing."""
        return {
            "rule": f"{self.condition['col']} {self.condition['op']} {self.condition['val']}" if self.condition else "Default",
            "conclusion": self.conclusion,
            "cornerstone_case": self.cornerstone.get('Id', 'Root') if self.cornerstone else "N/A",
            "if_true_exception": self.if_true.to_dict() if self.if_true else None,
            "if_false_alternative": self.if_false.to_dict() if self.if_false else None
        }

class AutoRDRLearner:
    def __init__(self, target_col):
        self.target_col = target_col
        self.root = None

    def classify(self, row):
        """Traverse the tree to find the most specific matching rule."""
        curr = self.root
        last_match = curr
        while curr:
            if curr.evaluate(row):
                last_match = curr
                curr = curr.if_true
            else:
                curr = curr.if_false
        return last_match

    def _induce_rule(self, new_row, cornerstone_row, features):
        """Finds the feature with the largest difference between two rows to create a rule."""
        best_col = None
        max_diff = -1
        for col in features:
            try:
                diff = abs(float(new_row[col]) - float(cornerstone_row[col]))
                if diff > max_diff:
                    max_diff = diff
                    best_col = col
            except: continue
        
        if best_col:
            v_new, v_old = float(new_row[best_col]), float(cornerstone_row[best_col])
            threshold = (v_new + v_old) / 2
            op = '>' if v_new > v_old else '<='
            return {'col': best_col, 'op': op, 'val': round(threshold, 3)}
        return {'col': features[0], 'op': '==', 'val': new_row[features[0]]}

    def train(self, df):
        features = [c for c in df.columns if c != self.target_col and c != 'Id']
        # Start with a default root based on the first record
        first = df.iloc[0]
        self.root = RDRNode(conclusion=first[self.target_col], cornerstone=first.to_dict())

        for _, row in df.iterrows():
            match_node = self.classify(row)
            if str(match_node.conclusion) != str(row[self.target_col]):
                # Create a new rule to fix this mistake
                new_cond = self._induce_rule(row, match_node.cornerstone, features)
                new_node = RDRNode(condition=new_cond, conclusion=row[self.target_col], cornerstone=row.to_dict())
                
                if match_node.evaluate(row):
                    if not match_node.if_true: match_node.if_true = new_node
                    else:
                        temp = match_node.if_true
                        while temp.if_false: temp = temp.if_false
                        temp.if_false = new_node
                else:
                    temp = match_node
                    while temp.if_false: temp = temp.if_false
                    temp.if_false = new_node

def main():
    parser = argparse.ArgumentParser(description="Automated RDR Rule Inducer")
    parser.add_argument("--input", required=True, help="Path to .xlsx or .csv file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="learned_rules.json", help="Output JSON file")
    args = parser.parse_args()

    # Smart File Loading
    try:
        ext = os.path.splitext(args.input)[1].lower()
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(args.input)
        else:
            df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error: Could not read file. {e}")
        return

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    learner = AutoRDRLearner(args.target)
    print(f"Building RDR tree from {len(train_df)} cases...")
    learner.train(train_df)

    with open(args.output, 'w') as f:
        json.dump(learner.root.to_dict(), f, indent=4)
    print(f"Success: Rules saved to {args.output}")

    # Testing
    y_true = test_df[args.target].astype(str)
    y_pred = [str(learner.classify(row).conclusion) for _, row in test_df.iterrows()]

    print("\n--- PERFORMANCE REPORT ---")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()

