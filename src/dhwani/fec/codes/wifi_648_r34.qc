# IEEE 802.11n LDPC matrix prototype, n = 648, rate 3/4, Z = 27
# rows cols Z, then the base matrix: -1 = Z x Z zero block, s >= 0 = Z x Z identity
# cyclically shifted right by s columns (row r of the block has its one in column (r + s) mod Z)
# source: https://mentor.ieee.org/802.11/dcn/05/11-05-1102-04-000n-joint-proposal-phy-specification.doc (Appendix B, Table B.1(c))
# source: https://www.mathworks.com/help/comm/ref/ldpcquasicyclicmatrix.html (identical)
6 24 27
16 17 22 24 9 3 14 -1 4 2 7 -1 26 -1 2 -1 21 -1 1 0 -1 -1 -1 -1
25 12 12 3 3 26 6 21 -1 15 22 -1 15 -1 4 -1 -1 16 -1 0 0 -1 -1 -1
25 18 26 16 22 23 9 -1 0 -1 4 -1 4 -1 8 23 11 -1 -1 -1 0 0 -1 -1
9 7 0 1 17 -1 -1 7 3 -1 3 23 -1 16 -1 -1 21 -1 0 -1 -1 0 0 -1
24 5 26 7 1 -1 -1 15 24 15 -1 8 -1 13 -1 13 -1 11 -1 -1 -1 -1 0 0
2 2 19 14 24 1 15 19 -1 21 -1 2 -1 24 -1 3 -1 2 1 -1 -1 -1 -1 0
