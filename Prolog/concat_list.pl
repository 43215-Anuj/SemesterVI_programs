concat([],L2,L2).
concat([Head|Tail],L2,[Head|L3]) :- concat(Tail,L2,L3).
