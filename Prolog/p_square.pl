init(F) :- F is 0.
down_up(V) :-
                    V is V+1,
                    write("*"),nl,
                    Inc =< H,
                    down_up(Inc,H,Step).

