def Get_Prompt(user_angles, user_lean, pro_angles, pro_lean, user_input):
    preface = """
    The following data represents key biomechanical measurements
    from a golfer's swing captured using pose estimation at different
    points in time presented as swing events, ie. address, top of
    backswing, impact, etc. Analyze these metrics to identify the 2-3
    most significant technical issues affecting this golfer's swing.
    Prioritize issues that would have the greatest impact on consistency
    and distance if corrected. To help you with this, you will also be
    provided with metrics from a professional golfer for comparison to 
    the user, and some context from the user about what issues they are
    experiencing.
    """
    
    # Add context that this is a conversation with the user:
    system_message = """
    You are a friendly and knowledgeable golf coach. Your task is to provide personalized feedback
    based on the user's golf swing metrics, explain the technical issues in a simple way, and suggest drills
    to help the user improve their swing for better distance and consistency.
    Always address the user directly and ensure your tone is encouraging and helpful.
    """
    
    # Add the metric description as before
    metric_background = """
    Here is some information about the specific metrics you will be analyzing:
    1. **Swing Angles**: These are the angles of the golfer's body and arms at different points in the swing (all in degrees).
        The angles for each time point are in order as follows: left arm - shoulders, right arm - shoulders, 
        shoulders - ground, hips - ground, upper left arm - lower left arm, upper right arm - lower right arm.
    2. **Lean**: This refers to the displacement of the midpoint of the golfer's hips compared to their starting
        position. This is used to determine how the golfer shifts their weight throughout the swing and is
        presented in arbitrary units. Note that proper weight transfer is indicated by progressive hip movement toward target during downswing phase.
    """
    
    user_metrics = f"""
    Here are the metrics for the user at each swing event:
    **Swing Angles**:
    1. Address: {user_angles['Address'][0]}, {user_angles['Address'][1]}, {user_angles['Address'][2]}, {user_angles['Address'][3]}, {user_angles['Address'][4]}, {user_angles['Address'][5]}
    2. Toe-up: {user_angles['Toe-up'][0]}, {user_angles['Toe-up'][1]}, {user_angles['Toe-up'][2]}, {user_angles['Toe-up'][3]}, {user_angles['Toe-up'][4]}, {user_angles['Toe-up'][5]}
    3. Mid-backswing: {user_angles['Mid-backswing'][0]}, {user_angles['Mid-backswing'][1]}, {user_angles['Mid-backswing'][2]}, {user_angles['Mid-backswing'][3]}, {user_angles['Mid-backswing'][4]}, {user_angles['Mid-backswing'][5]}
    4. Top: {user_angles['Top'][0]}, {user_angles['Top'][1]}, {user_angles['Top'][2]}, {user_angles['Top'][3]}, {user_angles['Top'][4]}, {user_angles['Top'][5]}
    5. Mid-downswing: {user_angles['Mid-downswing'][0]}, {user_angles['Mid-downswing'][1]}, {user_angles['Mid-downswing'][2]}, {user_angles['Mid-downswing'][3]}, {user_angles['Mid-downswing'][4]}, {user_angles['Mid-downswing'][5]}
    6. Impact: {user_angles['Impact'][0]}, {user_angles['Impact'][1]}, {user_angles['Impact'][2]}, {user_angles['Impact'][3]}, {user_angles['Impact'][4]}, {user_angles['Impact'][5]}
    **Lean**:
    1. Address: {user_lean['Address']}
    2. Toe-up: {user_lean['Toe-up']}
    3. Mid-backswing: {user_lean['Mid-backswing']}
    4. Top: {user_lean['Top']}
    5. Mid-downswing: {user_lean['Mid-downswing']}
    6. Impact: {user_lean['Impact']}
    """

    pro_metrics = f"""
    Here are the metrics for the professional golfer at each swing event:
    **Swing Angles**:
    1. Address: {pro_angles['Address'][0]}, {pro_angles['Address'][1]}, {pro_angles['Address'][2]}, {pro_angles['Address'][3]}, {pro_angles['Address'][4]}, {pro_angles['Address'][5]}
    2. Toe-up: {pro_angles['Toe-up'][0]}, {pro_angles['Toe-up'][1]}, {pro_angles['Toe-up'][2]}, {pro_angles['Toe-up'][3]}, {pro_angles['Toe-up'][4]}, {pro_angles['Toe-up'][5]}
    3. Mid-backswing: {pro_angles['Mid-backswing'][0]}, {pro_angles['Mid-backswing'][1]}, {pro_angles['Mid-backswing'][2]}, {pro_angles['Mid-backswing'][3]}, {pro_angles['Mid-backswing'][4]}, {pro_angles['Mid-backswing'][5]}
    4. Top: {pro_angles['Top'][0]}, {pro_angles['Top'][1]}, {pro_angles['Top'][2]}, {pro_angles['Top'][3]}, {pro_angles['Top'][4]}, {pro_angles['Top'][5]}
    5. Mid-downswing: {pro_angles['Mid-downswing'][0]}, {pro_angles['Mid-downswing'][1]}, {pro_angles['Mid-downswing'][2]}, {pro_angles['Mid-downswing'][3]}, {pro_angles['Mid-downswing'][4]}, {pro_angles['Mid-downswing'][5]}
    6. Impact: {pro_angles['Impact'][0]}, {pro_angles['Impact'][1]}, {pro_angles['Impact'][2]}, {pro_angles['Impact'][3]}, {pro_angles['Impact'][4]}, {pro_angles['Impact'][5]}
    **Lean**:
    1. Address: {pro_lean['Address']}
    2. Toe-up: {pro_lean['Toe-up']}
    3. Mid-backswing: {pro_lean['Mid-backswing']}
    4. Top: {pro_lean['Top']}
    5. Mid-downswing: {pro_lean['Mid-downswing']}
    6. Impact: {pro_lean['Impact']}
    """

    user_input = f"""
    Here is some context from the user:
    {user_input}
    """

    closing = """
    Provide feedback in the following format: (1) Key Findings, (2) Technical Explanation, (3) Recommendations and drills.
    Please address the user directly and offer advice that would help them improve their swing. Also, please make sure that
    all formatting in your response is in the form of html tags.
    """

    return system_message + preface + metric_background + user_metrics + pro_metrics + user_input + closing