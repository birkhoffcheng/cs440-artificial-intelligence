'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
	'''
	@param nonstandard_rules (dict) - dict from ruleIDs to rules
		Each rule is a dict:
		rule['antecedents'] contains the rule antecedents (a list of propositions)
		rule['consequent'] contains the rule consequent (a proposition).

	@return standardized_rules (dict) - an exact copy of nonstandard_rules,
		except that the antecedents and consequent of every rule have been changed
		to replace the word "something" with some variable name that is
		unique to the rule, and not shared by any other rule.
	@return variables (list) - a list of the variable names that were created.
		This list should contain only the variables that were used in rules.
	'''
	standardized_rules = copy.deepcopy(nonstandard_rules)
	ctr = 0
	variables = []
	for k in standardized_rules:
		need_standardize = False
		for anteced in standardized_rules[k]['antecedents']:
			if 'something' in anteced:
				need_standardize = True
		if 'something' in standardized_rules[k]['consequent']:
			need_standardize = True
		if need_standardize:
			variables.append('x' + str(ctr))
			for atidx in range(len(standardized_rules[k]['antecedents'])):
				for tokenidx in range(len(standardized_rules[k]['antecedents'][atidx])):
					if standardized_rules[k]['antecedents'][atidx][tokenidx] == 'something':
						standardized_rules[k]['antecedents'][atidx][tokenidx] = 'x' + str(ctr)
			for tokenidx in range(len(standardized_rules[k]['consequent'])):
				if standardized_rules[k]['consequent'][tokenidx] == 'something':
					standardized_rules[k]['consequent'][tokenidx] = 'x' + str(ctr)
			ctr += 1
	return standardized_rules, variables

def replace(l, elem_from, elem_to):
	for i in range(len(l)):
		if l[i] == elem_from:
			l[i] = elem_to

def unify(query, datum, variables):
	'''
	@param query: proposition that you're trying to match.
	  The input query should not be modified by this function; consider deepcopy.
	@param datum: proposition against which you're trying to match the query.
	  The input datum should not be modified by this function; consider deepcopy.
	@param variables: list of strings that should be considered variables.
	  All other strings should be considered constants.

	Unification succeeds if (1) every variable x in the unified query is replaced by a
	variable or constant from datum, which we call subs[x], and (2) for any variable y
	in datum that matches to a constant in query, which we call subs[y], then every
	instance of y in the unified query should be replaced by subs[y].

	@return unification (list): unified query, or None if unification fails.
	@return subs (dict): mapping from variables to values, or None if unification fails.
	   If unification is possible, then answer already has all copies of x replaced by
	   subs[x], thus the only reason to return subs is to help the calling function
	   to update other rules so that they obey the same substitutions.

	Examples:

	unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
	  unification = [ 'a', 'eats', 'b', False ]
	  subs = { "x":"a", "y":"b" }
	unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
	  unification = ['bobcat','eats','squirrel',True]
	  subs = { 'a':'bobcat', 'y':'squirrel' }
	unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
	  unification = ['a','eats','a',True]
	  subs = { 'x':'a' }
	unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
	  unification = ['bobcat','eats','bobcat',True],
	  subs = {'x':'a', 'a':'bobcat'}
	  When the 'x':'a' substitution is detected, the query is changed to
	  ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is
	  detected, the query is changed to ['bobcat','eats','bobcat',True], which
	  is the value returned as the answer.
	unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
	  unification = ['bobcat','eats','bobcat',True],
	  subs = {'a':'x', 'x':'bobcat'}
	  When the 'a':'x' substitution is detected, the query is changed to
	  ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution
	  is detected, the query is changed to ['bobcat','eats','bobcat',True], which is
	  the value returned as the answer.
	unify([...,True],[...,False],[...]) should always return None, None, regardless of the
	  rest of the contents of the query or datum.
	'''
	if query[1] != datum[1] or query[3] != datum[3]:
		return None, None
	subs = {}
	variables = set(variables)
	unification = copy.deepcopy(query)
	for i in range(len(unification)):
		if unification[i] == datum[i]:
			continue
		if unification[i] in variables:
			subs[unification[i]] = datum[i]
			replace(unification, unification[i], datum[i])
		elif datum[i] in variables:
			subs[datum[i]] = unification[i]
			replace(unification, datum[i], unification[i])
		else:
			return None, None
	return unification, subs

def apply(rule, goals, variables):
	'''
	@param rule: A rule that is being tested to see if it can be applied
	  This function should not modify rule; consider deepcopy.
	@param goals: A list of propositions against which the rule's consequent will be tested
	  This function should not modify goals; consider deepcopy.
	@param variables: list of strings that should be treated as variables

	Rule application succeeds if the rule's consequent can be unified with any one of the goals.

	@return applications: a list, possibly empty, of the rule applications that
	   are possible against the present set of goals.
	   Each rule application is a copy of the rule, but with both the antecedents
	   and the consequent modified using the variable substitutions that were
	   necessary to unify it to one of the goals. Note that this might require
	   multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
	   based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
	   The length of the applications list is 0 <= len(applications) <= len(goals).
	   If every one of the goals can be unified with the rule consequent, then
	   len(applications)==len(goals); if none of them can, then len(applications)=0.
	@return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
	   goalsets[i] is a copy of goals (a list) in which the goal that unified with
	   applications[i]['consequent'] has been removed, and replaced by
	   the members of applications[i]['antecedents'].

	Example:
	rule={
	  'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
	  'consequent':['x','eats','squirrel',False]
	}
	goals=[
	  ['bobcat','eats','squirrel',False],
	  ['bobcat','visits','squirrel',True],
	  ['bald eagle','eats','squirrel',False]
	]
	variables=['x','y','a','b']

	applications, newgoals = submitted.apply(rule, goals, variables)

	applications==[
	  {
		'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
		'consequent':['bobcat','eats','squirrel',False]
	  },
	  {
		'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
		'consequent':['bald eagle','eats','squirrel',False]
	  }
	]
	newgoals==[
	  [
		['bobcat','visits','squirrel',True],
		['bald eagle','eats','squirrel',False]
		['bobcat','is','nice',True],
		['bobcat','is','hungry',False]
	  ],[
		['bobcat','eats','squirrel',False]
		['bobcat','visits','squirrel',True],
		['bald eagle','is','nice',True],
		['bald eagle','is','hungry',False]
	  ]
	'''
	applications = []
	goalsets = []
	for goal in goals:
		unification, subs = unify(goal, rule['consequent'], variables)
		if unification is None:
			continue
		application = {'consequent': unification}
		antecedents = copy.deepcopy(rule['antecedents'])
		for anteidx in range(len(antecedents)):
			for tokidx in range(len(antecedents[anteidx])):
				if subs.get(antecedents[anteidx][tokidx], None) is not None:
					antecedents[anteidx][tokidx] = subs[antecedents[anteidx][tokidx]]
		application['antecedents'] = antecedents
		applications.append(application)
		goalset = list(goals)
		goalset.remove(goal)
		goalset.extend(antecedents)
		goalsets.append(goalset)
	return applications, goalsets

def convert_to_key(goals):
	goals = copy.deepcopy(goals)
	for i in range(len(goals)):
		goals[i] = tuple(goals[i])
	return frozenset(goals)

def backward_chain(query, rules, variables):
	'''
	@param query: a proposition, you want to know if it is true
	@param rules: dict mapping from ruleIDs to rules
	@param variables: list of strings that should be treated as variables

	@return proof (list): a list of rule applications
	  that, when read in sequence, conclude by proving the truth of the query.
	  If no proof of the query was found, you should return proof=None.
	'''
	q = queue.SimpleQueue()
	proof_chains = {convert_to_key([query]): []}
	q.put([query])
	while q.qsize() > 0:
		node = q.get()
		node_chain = proof_chains[convert_to_key(node)]
		for k in rules:
			applications, goalsets = apply(rules[k], node, variables)
			for i in range(len(applications)):
				if len(goalsets[i]) == 0:
					return [applications[i]] + node_chain
				if convert_to_key(goalsets[i]) in proof_chains:
					continue
				proof_chains[convert_to_key(goalsets[i])] = [applications[i]] + node_chain
				q.put(goalsets[i])
	return None
