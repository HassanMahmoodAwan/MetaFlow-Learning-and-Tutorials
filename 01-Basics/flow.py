from metaflow import FlowSpec, step, Parameter, JSONType, card


class BasicFlow(FlowSpec):
    
    all_users = Parameter("users", help="interest_rate for loan", required=True, type=JSONType, default=[("Hassan", 6), ("Ali", 9), ("Haris", 18)])
    
    @card
    @step
    def start(self):
        """
            Start of Workflow
        """
        print("Workflow Started! ")
        print("Users are : ", self.all_users)
        self.next(self.users)
    
    @card
    @step
    def users(self):
        self.users_list = self.all_users
        self.next(self.create_artifacts, foreach="users_list")    
    
    @card
    @step
    def create_artifacts(self):
        """
            Defining and Initializing Artifacts 
            => For Loan.
        """
        self.name = self.input[0]
        self.interestRate = self.input[1]
        print(self.name)
        print(self.interestRate)
        
        self.principal = 20000000
        self.period = 5
        
        self.next(self.compute_payable, self.actual_monthly_installment)
    
    @card
    @step
    def compute_payable(self):
        self.payable_Amount = float(self.principal * self.interestRate * self.period) / 100
        self.next(self.join)
    
    @card
    @step
    def actual_monthly_installment(self):
        self.installment = self.principal / (self.period * 12)
        self.next(self.join)
    
    @card
    @step
    def join(self, inputs):
        """
            Join Takes Global parameters, but not any artifacts like principal, period
        """
        self.payable_Amount = inputs.compute_payable.payable_Amount
        self.installment = inputs.actual_monthly_installment.installment
        
        print(self.payable_Amount)
        print(self.installment)
        self.next(self.user_join)
     
    @card
    @step
    def user_join(self, inputs):
        print(self.all_users)
        self.next(self.end)      
    
    @card
    @step
    def end(self):
        """
            End of Workflow
        """
        print("Workflow Ended!")

if __name__ == "__main__":
    BasicFlow()
