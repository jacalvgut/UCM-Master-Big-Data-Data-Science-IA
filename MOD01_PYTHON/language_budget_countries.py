# -*- coding: utf-8 -*-
"""
PARTE F: Un cálculo masivo con Map-Reduce
"""

from mrjob.job import MRJob

class MRMovieBudget(MRJob):
    
    def mapper(self, _, line):
        
        row = line.split('|')
        
        language = row[2].strip()
        country = row[3].strip()
        
        try:
            budget = float(row[4]) if row[4].strip() else -1
        except ValueError:
            budget = -1
        
        if language and country and budget > 0:
        
            yield language, (country, budget)
        
    def reducer(self, language, values):
        countries = set()
        total_budget = 0.
        
        for country, budget in values:
            countries.add(country)
            total_budget += budget
            
        yield language, f'[{list(countries)},{round(total_budget)}]'
        
if __name__ == '__main__':
    MRMovieBudget.run()    