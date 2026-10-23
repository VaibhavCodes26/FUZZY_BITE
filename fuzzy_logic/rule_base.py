from skfuzzy import control as ctrl

def build_rule_base(variables):
    """
    Constructs the fuzzy rule base using defined variables.
    """
    R = variables['rating']
    P = variables['price_suitability']
    D = variables['distance']
    C = variables['cuisine_match']
    S = variables['suitability']

    rules = [
        # Very High Rules
        ctrl.Rule(R['High'] & P['Good'] & D['Near'] & C['Excellent'], S['Very High'], label="Rule_VH1"),
        
        # High Rules
        ctrl.Rule(R['High'] & P['Moderate'] & C['Excellent'], S['High'], label="Rule_H1"),
        ctrl.Rule(R['Average'] & P['Good'] & D['Near'], S['High'], label="Rule_H2"),
        ctrl.Rule(R['High'] & C['Partial'] & D['Near'], S['High'], label="Rule_H3"),
        
        # Moderate Rules
        ctrl.Rule(R['Average'] & P['Moderate'] & D['Moderate'], S['Moderate'], label="Rule_M1"),
        ctrl.Rule(R['High'] & P['Poor'] & C['Excellent'], S['Moderate'], label="Rule_M2"),
        ctrl.Rule(R['Average'] & C['Partial'] & D['Moderate'], S['Moderate'], label="Rule_M3"),
        
        # Low Rules
        ctrl.Rule(P['Poor'] & D['Far'], S['Low'], label="Rule_L1"),
        ctrl.Rule(R['Average'] & P['Poor'] & C['Poor'], S['Low'], label="Rule_L2"),
        
        # Very Low Rules
        ctrl.Rule(R['Low'] | C['Poor'], S['Very Low'], label="Rule_VL1"),
        ctrl.Rule(R['Low'] & P['Poor'] & D['Far'], S['Very Low'], label="Rule_VL2")
    ]
    
    return rules