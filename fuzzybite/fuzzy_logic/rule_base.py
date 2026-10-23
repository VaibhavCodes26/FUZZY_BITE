from skfuzzy import control as ctrl

def build_rule_base(variables):
    R, P, D, C, S = variables['rating'], variables['price_suitability'], variables['distance'], variables['cuisine_match'], variables['suitability']
    rules = [
        ctrl.Rule(R['High'] & P['Good'] & D['Near'] & C['Excellent'], S['Very High']),
        ctrl.Rule(R['High'] & P['Moderate'] & C['Excellent'], S['High']),
        ctrl.Rule(R['Average'] & P['Good'] & D['Near'], S['High']),
        ctrl.Rule(R['High'] & C['Partial'] & D['Near'], S['High']),
        ctrl.Rule(R['Average'] & P['Moderate'] & D['Moderate'], S['Moderate']),
        ctrl.Rule(R['High'] & P['Poor'] & C['Excellent'], S['Moderate']),
        ctrl.Rule(R['Average'] & C['Partial'] & D['Moderate'], S['Moderate']),
        ctrl.Rule(P['Poor'] & D['Far'], S['Low']),
        ctrl.Rule(R['Average'] & P['Poor'] & C['Poor'], S['Low']),
        ctrl.Rule(R['Low'] | C['Poor'], S['Very Low']),
        ctrl.Rule(R['Low'] & P['Poor'] & D['Far'], S['Very Low'])
    ]
    return rules
