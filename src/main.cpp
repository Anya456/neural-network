#include <iostream>
#include <vector>
#include "Matrix.h"

using namespace std;

int main()
{
    int r1, r2, c1, c2;
    cin >> r1 >> c1;
    Matrix a(r1, c1);

    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c1; j++)
        {
            cin >> a(i, j);
        }
    }

    Matrix c = a * 4;
    c.print();
}