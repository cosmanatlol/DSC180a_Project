class simulation:
    def __init__(self, initial_price, initial_investment):
        self.price = initial_price
        self.shares = initial_investment/initial_price
        self.extra_investment = 0

    def invest_strat(self, buy):
        if self.price*self.shares + buy < 0:
            buy = -1*self.price*self.shares
        self.extra_investment += buy
        self.shares += buy/self.price
    
    def new_price(self, new_price):
        self.price = new_price

    def cur_val(self):
        return self.price*self.shares 
