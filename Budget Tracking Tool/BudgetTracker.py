import csv
import os

DATA_FILE = 'budget_data.csv'

def initialize_data_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type', 'Category', 'Amount', 'Description'])  # TODO: Add a timestamp field to the CSV header.

def add_transaction(transaction_type, category, amount, description):
    # TODO: Validate that the amount is a positive number.
    if ((type(amount) != float and type(amount) != int) or amount < 0):  #checks if the datatype of amount is not int or float and then checks if entered amount is negative
        print("Invalid amount")
    # TODO: Automatically add the current date and time when a transaction is recorded.
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([transaction_type, category, amount, description])

def update_transaction(index, transaction_type, category, amount, description):
    # TODO: Validate that the provided index is valid.
    total=read_transactions()
    if len(total)==0:
        print("no transaction exists to be deleted")
        return
    if index<0 or index>=len(total):
        print("invalid index....please enter a valid index")
        return
    temp_file = DATA_FILE + '.tmp'
    with open(DATA_FILE, mode='r') as file, open(temp_file, mode='w', newline='') as temp:
        reader = csv.reader(file)
        writer = csv.writer(temp)
        for i, row in enumerate(reader):
            if i == index + 1:  # To skip header
                writer.writerow([transaction_type, category, amount, description])
            else:
                writer.writerow(row)
    os.replace(temp_file, DATA_FILE)

def delete_transaction(index):
    # TODO: Validate that the provided index is valid.
    total=read_transactions()
    if len(total)==0:
        print("no transaction exists to be deleted")
        return
    if index<0 or index>=len(total):
        print("invalid index....please enter a valid index")
        return
    temp_file = DATA_FILE + '.tmp'
    with open(DATA_FILE, mode='r') as file, open(temp_file, mode='w', newline='') as temp:
        reader = csv.reader(file)
        writer = csv.writer(temp)
        for i, row in enumerate(reader):
            if i != index + 1:  
                writer.writerow(row)
    os.replace(temp_file, DATA_FILE)

def read_transactions():
    with open(DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        return list(reader)[1:]  

def generate_report():
    # TODO: Extend the report to show income and expense totals by category.
    # TODO: Add functionality to generate a summary of transactions within a specific date range.
    income = 0.0
    expenses = 0.0
    with open(DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            amount = float(row[2])
            if row[0] == 'income':
                income += amount
            elif row[0] == 'expense':
                expenses += amount
    savings = income - expenses

    print(f"Total Income: ${income:.2f}")
    print(f"Total Expenses: ${expenses:.2f}")
    print(f"Savings: ${savings:.2f}")

def export_to_excel_or_pdf():
    # TODO: Provide an option to export the transaction data to a PDF or Excel file.
    pass

def search_transactions():
    # TODO: Allow users to search for transactions by category, type, or description.
    pass

def undo_last_action():
    # TODO: Implement an "Undo" feature to revert the most recent addition, update, or deletion of a transaction.
    pass

def clear_screen():
    # TODO: Add functionality to clear the terminal screen at the start of each menu display.
    pass

def main():
    initialize_data_file()
    while True:
        # TODO: Call clear_screen() here for better readability.
        print("\nBudget Tracker")
        print("1. Add Transaction")
        print("2. Update Transaction")
        print("3. Delete Transaction")
        print("4. Generate Report")
        print("5. Search Transactions")  # TODO: Add this menu option for searching transactions.
        print("6. Export Data")  # TODO: Add this menu option for exporting data.
        print("7. Undo Last Action")  # TODO: Add this menu option for the undo feature.
        print("8. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            transaction_type = input("Enter transaction type (income/expense): ")
            category = input("Enter category: ")
            amount = input("Enter amount: ")
            description = input("Enter description (optional): ") or "No description"  # TODO: Handle optional descriptions.
            add_transaction(transaction_type, category, amount, description)
        elif choice == '2':
            total=read_transactions()
            if(len(total)==0):
                print("no transactions made yet..please add transactions first")
                continue
            else:
                 while True:
                     index = int(input("Enter transaction index to update: "))   # TODO: Use a validated index.
                     if(index>=0 and index<len(total)):
                         break
                     else:
                         print("Invalid index..please enter valid index")

            transaction_type = input("Enter transaction type (income/expense): ")
            category = input("Enter category: ")
            amount = input("Enter amount: ")
            description = input("Enter description (optional): ") 
            update_transaction(index, transaction_type, category, amount, description)
        elif choice == '3':
            total=read_transactions()
            if(len(total)==0):
                print("no transactions to delete..please first add transactions")
                continue
            else:
                while True:
                    index = int(input("Enter transaction index to delete: "))    # TODO: Use a validated index.
                    if index>=0 and index<len(total):
                        break
                    else:
                        print("invalid index...please enter again")
                delete_transaction(index)
        elif choice == '4':
            generate_report()
        elif choice == '5':
            search_transactions()  # TODO: Implement this function.
        elif choice == '6':
            export_to_excel_or_pdf()  # TODO: Implement this function.
        elif choice == '7':
            undo_last_action()  # TODO: Implement this function.
        elif choice == '8':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
