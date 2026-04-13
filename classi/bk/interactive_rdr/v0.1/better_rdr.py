import pandas as pd
import json
import argparse
import os
import sys

class RDRNode:
    def __init__(self, condition=None, conclusion=None, case=None):
        """
        SCRDR Node structure:
        - condition: A dictionary with 'col', 'op', 'val'
        - conclusion: The target label
        - case: The 'Cornerstone Case' that prompted this rule
        - if_true: Exception/Refinement branch (If parent is true but conclusion is wrong)
        - if_false: Alternative/Next branch (If parent is false, try this next)
        """
        self.condition = condition
        self.conclusion = conclusion
        self.case = case
        self.if_true = None
        self.if_false = None

    def evaluate(self, row):
        if not self.condition:
            return True
        
        col, op, val = self.condition['col'], self.condition['op'], self.condition['val']
        row_val = row[col]
        
        try:
            # Handle numeric vs string comparison
            if isinstance(row_val, (int, float)) and not isinstance(val, bool):
                val = float(val)
            
            if op == '==': return str(row_val) == str(val)
            if op == '!=': return str(row_val) != str(val)
            if op == '<':  return float(row_val) < float(val)
            if op == '>':  float(row_val) > float(val)
            if op == '<=': return float(row_val) <= float(val)
            if op == '>=': return float(row_val) >= float(val)
        except Exception:
            return str(row_val) == str(val)
        return False

    def to_dict(self):
        return {
            'condition': self.condition,
            'conclusion': self.conclusion,
            'case': self.case,
            'if_true': self.if_true.to_dict() if self.if_true else None,
            'if_false': self.if_false.to_dict() if self.if_false else None
        }

    @classmethod
    def from_dict(cls, data):
        if data is None: return None
        node = cls(data['condition'], data['conclusion'], data['case'])
        node.if_true = cls.from_dict(data['if_true'])
        node.if_false = cls.from_dict(data['if_false'])
        return node

class SCRDR_Engine:
    def __init__(self, target_col):
        self.root = RDRNode(conclusion="Default") # Default root
        self.target_col = target_col
        self.cornerstone_cases = []

    def classify(self, row):
        """Traverse the tree to find the conclusion."""
        curr = self.root
        last_match = curr
        
        while curr:
            if curr.evaluate(row):
                last_match = curr
                curr = curr.if_true # Move to exceptions
            else:
                curr = curr.if_false # Move to next alternative
        return last_match

    def add_rule(self, row, predicted_node):
        print("\n" + "="*50)
        print("KNOWLEDGE ACQUISITION MODE")
        print(f"Current Row:\n{row.to_dict()}")
        print(f"System Predicted: '{predicted_node.conclusion}'")
        print(f"Actual Correct:   '{row[self.target_col]}'")
        print("="*50)

        # 1. Select Feature
        cols = [c for c in row.index if c != self.target_col]
        for i, col in enumerate(cols):
            print(f"{i+1}: {col} (Value: {row[col]})")
        
        while True:
            try:
                choice = int(input("\nSelect feature serial number: "))
                selected_col = cols[choice-1]
                break
            except (ValueError, IndexError):
                print("Invalid selection. Please enter a number from the list.")

        # 2. Select Operator
        ops = ['==', '!=', '<', '>', '<=', '>=']
        for i, op in enumerate(ops):
            print(f"{i+1}: {op}")
        
        while True:
            try:
                op_choice = int(input("Select operator number: "))
                selected_op = ops[op_choice-1]
                break
            except (ValueError, IndexError):
                print("Invalid operator choice.")

        # 3. Value
        val = input(f"Enter threshold value (Current is {row[selected_col]}): ")
        
        new_cond = {'col': selected_col, 'op': selected_op, 'val': val}
        new_node = RDRNode(condition=new_cond, conclusion=row[self.target_col], case=row.to_dict())

        # Determine where to attach
        # If the predicted node matched (but was wrong), add as Exception (if_true)
        # Otherwise, add as Alternative (if_false)
        if predicted_node.evaluate(row):
            if predicted_node.if_true:
                # Find the end of the alternative chain for this exception
                temp = predicted_node.if_true
                while temp.if_false: temp = temp.if_false
                temp.if_false = new_node
            else:
                predicted_node.if_true = new_node
        else:
            if predicted_node.if_false:
                temp = predicted_node.if_false
                while temp.if_false: temp = temp.if_false
                temp.if_false = new_node
            else:
                predicted_node.if_false = new_node
        
        print("Rule added successfully.")

def main():
    parser = argparse.ArgumentParser(description="Interactive SCRDR Rule Builder")
    parser.add_argument("--input", required=True, help="Path to CSV or Excel file")
    parser.add_argument("--target", required=True, help="Column name to predict")
    parser.add_argument("--tree", default="kb_tree.json", help="File to save/load rules")
    parser.add_argument("--sheet", default=0, help="Excel sheet index/name")
    args = parser.parse_args()

    # Load Data
    try:
        if args.input.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.input, sheet_name=args.sheet)
        else:
            df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Load Tree
    engine = SCRDR_Engine(args.target)
    if os.path.exists(args.tree):
        with open(args.tree, 'r') as f:
            engine.root = RDRNode.from_dict(json.load(f))
        print(f"Loaded existing Knowledge Base from {args.tree}")

    print("\n--- Starting Interactive Classification ---")
    print("Guide: For each row, the system will attempt to predict the label.")
    print("If it's wrong, you will be prompted to define a refinement rule.")
    
    for idx, row in df.iterrows():
        pred_node = engine.classify(row)
        
        if str(pred_node.conclusion) != str(row[args.target]):
            print(f"\n[Row {idx}] Mismatch found!")
            engine.add_rule(row, pred_node)
            
            # Save progress after every rule addition
            with open(args.tree, 'w') as f:
                json.dump(engine.root.to_dict(), f, indent=2)
        else:
            # Silent if correct to speed up processing
            pass

    print("\nAll rows processed. Knowledge Base saved to:", args.tree)

if __name__ == "__main__":
    main()

