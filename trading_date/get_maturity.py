# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 11. 26
"""
import calendar
from datetime import datetime


def meetup_day(year, month, weekday, spec_weekday):
    last_day = calendar.monthrange(year, month)[1]
    wkday = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    schedule_day = wkday[weekday]

    if spec_weekday == "teenth":
        check_range = range(13, 20)
    elif spec_weekday == "last":
        check_range = range(last_day - 6, last_day + 1)
    else:
        spec_weekday = int(spec_weekday[0:1])
        check_range = range(7 * spec_weekday - 6, 7 * spec_weekday + 1)

    for index in check_range:
        if index > last_day:
            break
        if schedule_day == calendar.weekday(year, month, index):
            schedule_day = index

    return datetime(year, month, schedule_day)


if __name__ == "__main__":
    meetup_day(2014, 3, "Thursday", "2nd")
    meetup_day(2013, 6, "Wednesday", "4th")
    meetup_day(2013, 12, "Monday", "1st")
    meetup_day(2015, 5, "Tuesday", "teenth")
    meetup_day(2015, 4, "Thursday", "last")
